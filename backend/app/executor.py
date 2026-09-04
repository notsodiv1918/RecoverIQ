import json
import random
import datetime as dt
from sqlalchemy.orm import Session

from . import config, models, policy, simulator, diagnosis, bandit, razorpay_client
from .detector import severity_score


def _log_audit(db: Session, txn_id: int, stage: str, payload: dict):
    ev = models.AuditEvent(
        transaction_id=txn_id,
        stage=stage,
        payload=json.dumps(payload, default=str),
    )
    db.add(ev)


def _in_quiet_hours(hour: int) -> bool:
    start, end = config.QUIET_HOURS_START, config.QUIET_HOURS_END
    if start > end:  # wraps midnight, e.g. 22 -> 8
        return hour >= start or hour < end
    return start <= hour < end


def apply_guardrails(txn, proposed_action: str, proposed_channel: str, current_batch: int) -> tuple[str, str, str]:
    """Returns (final_action, final_channel, guardrail_note). May override
    the policy's proposed action entirely."""

    # 1. Dispute pending -> always human, no automation, regardless of attempt count
    if txn.root_cause == "dispute_pending" and config.DISPUTE_AUTO_ESCALATE:
        return "escalate_human_review", "human_agent", "compliance: dispute_pending always routes to human"

    # 2. Max attempts hard cap
    if txn.attempt_count >= config.MAX_ATTEMPTS:
        return "mark_dead", "none", f"max_attempts ({config.MAX_ATTEMPTS}) reached"

    # 3. Amount threshold -> requires human approval, auto-actions blocked
    if txn.amount > config.HUMAN_APPROVAL_THRESHOLD and proposed_action not in (
        "escalate_human_review", "mark_dead"
    ):
        return "escalate_human_review", "human_agent", (
            f"amount {txn.amount} exceeds auto-approval threshold "
            f"({config.HUMAN_APPROVAL_THRESHOLD}); human sign-off required"
        )

    # 4. Cooldown between touches on same transaction (measured in batch runs)
    if txn.last_action_batch is not None:
        gap = current_batch - txn.last_action_batch
        if gap < config.COOLDOWN_BATCHES:
            return "no_action_cooldown", "none", (
                f"only {gap} batch run(s) since last touch, cooldown requires "
                f"{config.COOLDOWN_BATCHES}"
            )

    # 5. Opt-out / DND — block customer-contact channels, allow silent actions
    customer = txn.customer
    contact_actions = {"send_reminder", "send_discount_coupon", "send_payment_link", "escalate_voice_call"}
    if customer and (customer.opt_out or customer.dnd) and proposed_action in contact_actions:
        if proposed_action == "escalate_voice_call" and customer.dnd:
            return "escalate_human_review", "human_agent", "DND active: voice call blocked, routed to human"
        if customer.opt_out:
            return "escalate_human_review", "human_agent", "customer opted out of automated contact"

    return proposed_action, proposed_channel, "no override — guardrails clear"


def process_transaction(db: Session, txn: models.Transaction, current_batch: int):
    """Runs one transaction through detect -> diagnose -> policy -> guardrails
    -> execute -> simulate outcome -> audit. Idempotent-ish: safe to call once
    per batch pass per transaction."""

    # ---- 1. Detect / prioritize ----
    sev = severity_score(txn)
    _log_audit(db, txn.id, "detect", {"severity": sev, "type": txn.type, "amount": txn.amount})

    # ---- 2. Diagnose (only once, cached on the transaction) ----
    if not txn.root_cause:
        root_cause, confidence, rationale = diagnosis.diagnose(txn)
        txn.root_cause = root_cause
        txn.diagnosis_confidence = confidence
        _log_audit(db, txn.id, "diagnose", {
            "root_cause": root_cause, "confidence": confidence, "rationale": rationale,
        })

    # ---- 3. Policy proposes an action ----
    # Thompson-sampling bandit by default (learns from real outcomes as the
    # batch runs — see bandit.py). Falls back to the static ladder in
    # policy.py if USE_BANDIT_POLICY is off, e.g. for the policy-comparison
    # endpoint's baseline arm.
    if config.USE_BANDIT_POLICY:
        proposed_action, proposed_channel, policy_rationale = bandit.choose_action(db, txn, txn.root_cause)
    else:
        proposed_action, proposed_channel, policy_rationale = policy.decide_action(txn, txn.root_cause)
    _log_audit(db, txn.id, "policy_decision", {
        "proposed_action": proposed_action, "proposed_channel": proposed_channel,
        "rationale": policy_rationale,
    })

    # ---- 4. Guardrails may override ----
    final_action, final_channel, guardrail_note = apply_guardrails(
        txn, proposed_action, proposed_channel, current_batch
    )
    _log_audit(db, txn.id, "guardrail", {
        "final_action": final_action, "final_channel": final_channel, "note": guardrail_note,
    })

    # ---- 5. Simulated quiet-hours check (only for actions that contact the customer) ----
    sim_hour = random.randint(0, 23)
    contact_actions = {"send_reminder", "send_discount_coupon", "send_payment_link", "escalate_voice_call"}
    if final_action in contact_actions and _in_quiet_hours(sim_hour):
        log = models.ActionLog(
            transaction_id=txn.id, action_type=final_action, channel=final_channel,
            rationale=f"{policy_rationale} | {guardrail_note}",
            outcome="deferred_quiet_hours", recovered_amount=0.0, sim_hour=sim_hour,
        )
        db.add(log)
        _log_audit(db, txn.id, "action", {"action": final_action, "outcome": "deferred_quiet_hours", "sim_hour": sim_hour})
        txn.status = "in_progress"
        db.commit()
        return

    # ---- 6. Execute + simulate outcome ----
    link_info = None
    if final_action == "send_payment_link":
        # The one real external call in this pipeline: create an actual
        # Razorpay test-mode Payment Link (or a clearly-labeled simulated
        # one if no keys are configured -- see razorpay_client.py).
        link_info = razorpay_client.create_payment_link(txn)
        _log_audit(db, txn.id, "razorpay_link", link_info)

    if final_action == "mark_dead":
        recovered, amount = False, 0.0
        outcome = "dead"
    elif final_action == "no_action_cooldown":
        recovered, amount = False, 0.0
        outcome = "cooldown"
    elif final_action == "escalate_human_review":
        # human actions aren't auto-resolved by the simulator; still has a chance,
        # human agents historically recover a portion of escalated cases
        recovered, amount = simulator.simulate_outcome(txn, final_action)
        outcome = "recovered" if recovered else "pending_human_review"
        bandit.record_outcome(db, txn.root_cause, final_action, recovered)
    else:
        recovered, amount = simulator.simulate_outcome(txn, final_action)
        outcome = "recovered" if recovered else "no_response"
        bandit.record_outcome(db, txn.root_cause, final_action, recovered)

    log = models.ActionLog(
        transaction_id=txn.id, action_type=final_action, channel=final_channel,
        rationale=f"{policy_rationale} | {guardrail_note}",
        outcome=outcome, recovered_amount=amount, sim_hour=sim_hour,
        link_url=(link_info["link_url"] if link_info else None),
        link_is_real=(link_info["real"] if link_info else None),
    )
    db.add(log)
    _log_audit(db, txn.id, "outcome", {"outcome": outcome, "recovered_amount": amount})

    # ---- 7. Update transaction state ----
    if final_action not in ("no_action_cooldown",):
        txn.attempt_count += 1
        txn.last_action_batch = current_batch

    if recovered:
        txn.status = "recovered"
        txn.recovered_amount = amount
    elif final_action == "mark_dead":
        txn.status = "dead"
    elif final_action == "escalate_human_review":
        txn.status = "escalated"
    elif final_action == "no_action_cooldown":
        pass  # status stays as-is
    elif txn.attempt_count >= config.MAX_ATTEMPTS:
        txn.status = "dead"
    else:
        txn.status = "in_progress"

    db.commit()