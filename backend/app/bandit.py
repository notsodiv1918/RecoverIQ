"""
The actual "learning" system. Instead of a fixed lookup table, each
(root_cause, action) pair has a Beta(alpha, beta) posterior over its recovery
probability. Every batch run:
  1. samples a value from each candidate action's posterior (Thompson Sampling)
  2. picks whichever action sampled highest
  3. after the real (simulated) outcome comes back, updates that pair's
     alpha (success) or beta (failure)

Early on, posteriors are wide (prior = Beta(1,1), i.e. uniform) so choices are
essentially random across candidates — this IS the A/B test. As outcomes
accumulate, the posterior for a genuinely better action tightens around a
higher value and gets sampled highest more consistently — this IS the
learning. Nothing here is scripted to converge to a particular answer; it's
driven entirely by the outcomes simulator.py actually returns.

policy.py's static ACTION_LADDER is kept as-is and used as the "baseline
strategy" for the policy comparison endpoint (see /api/policy-comparison) —
a real empirical A vs B, not an assumed industry number.
"""
import random
from sqlalchemy.orm import Session
from . import models
from .policy import CHANNEL_BY_TIER, VOICE_ELIGIBLE_TYPES

# root_cause -> candidate actions the bandit is allowed to choose between.
# "mark_dead" is deliberately excluded — giving up is a guardrail stopping
# rule (max attempts), not a strategy the policy learns to prefer.
CANDIDATE_ACTIONS = {
    "card_expired":           ["send_payment_link", "escalate_human_review"],
    "insufficient_funds":     ["retry_payment", "send_reminder", "escalate_human_review"],
    "auth_friction":          ["send_payment_link", "retry_payment", "escalate_human_review"],
    "bank_decline_other":     ["retry_payment", "send_reminder", "escalate_human_review"],
    "fraud_review_needed":    ["escalate_human_review"],

    "price_sensitivity":      ["send_discount_coupon", "send_reminder"],
    "missing_payment_method": ["send_payment_link", "send_reminder"],
    "technical_friction":     ["send_reminder", "send_payment_link"],
    "low_intent":             ["send_reminder"],

    "forgot":                 ["send_reminder", "escalate_voice_call"],
    "cash_flow_issue":        ["send_reminder", "send_payment_link", "escalate_voice_call"],
    "dispute_pending":        ["escalate_human_review"],
    "invoice_error":          ["escalate_human_review"],
    "awaiting_approval":      ["send_reminder", "escalate_voice_call", "escalate_human_review"],

    "unknown":                ["send_reminder", "escalate_human_review"],
}


def _get_or_create_stat(db: Session, root_cause: str, action: str) -> models.PolicyStat:
    stat = (
        db.query(models.PolicyStat)
        .filter(models.PolicyStat.root_cause == root_cause, models.PolicyStat.action == action)
        .first()
    )
    if not stat:
        stat = models.PolicyStat(root_cause=root_cause, action=action, alpha=1.0, beta=1.0, times_chosen=0)
        db.add(stat)
        db.commit()
        db.refresh(stat)
    return stat


def _channel_for(action: str, txn) -> str:
    tier = txn.customer.tier if txn.customer else "retail"
    if action == "retry_payment":
        return "gateway"
    if action == "escalate_human_review":
        return "human_agent"
    if action == "escalate_voice_call":
        return "voice"
    return CHANNEL_BY_TIER.get(tier, "sms")


def choose_action(db: Session, txn, root_cause: str) -> tuple[str, str, str]:
    """Returns (action, channel, rationale). Rationale includes the full
    sampled posterior snapshot so the audit trail can show exactly why this
    action won over its alternatives — this is also what powers the
    'counterfactual' view (what would the OTHER action's learned odds have been)."""
    candidates = CANDIDATE_ACTIONS.get(root_cause, CANDIDATE_ACTIONS["unknown"])

    samples = {}
    snapshot = {}
    for action in candidates:
        stat = _get_or_create_stat(db, root_cause, action)
        samples[action] = random.betavariate(stat.alpha, stat.beta)
        snapshot[action] = {
            "learned_win_rate": round(stat.alpha / (stat.alpha + stat.beta), 3),
            "n_observations": int(stat.alpha + stat.beta - 2),
            "sampled_value": round(samples[action], 3),
        }

    chosen = max(samples, key=samples.get)

    if chosen == "escalate_voice_call" and txn.type not in VOICE_ELIGIBLE_TYPES:
        chosen = "send_reminder"

    channel = _channel_for(chosen, txn)

    rationale = (
        f"bandit proposes '{chosen}' via {channel} for root_cause={root_cause} "
        f"| candidates & learned posteriors: {snapshot}"
    )
    return chosen, channel, rationale


def record_outcome(db: Session, root_cause: str, action: str, recovered: bool):
    """The actual learning step — called once per real EXECUTED action with a
    genuine outcome (not for deferrals, cooldowns, or mark_dead). Deliberately
    keyed to the action that actually ran, which may differ from what the
    bandit originally proposed if a guardrail overrode it (e.g. DND blocking
    a voice call, or the amount threshold forcing human review) — this keeps
    times_chosen and n_observations consistent with each other, both counting
    real executions rather than raw proposals."""
    stat = _get_or_create_stat(db, root_cause, action)
    stat.times_chosen += 1
    if recovered:
        stat.alpha += 1.0
    else:
        stat.beta += 1.0
    db.commit()
