"""
Policy engine: pure Python decision table, NOT left to the LLM.
Given (root_cause, attempt_count, tier), returns the preferred action + channel.

Guardrails (max attempts, cooldown, amount threshold, quiet hours, opt-out/dnd,
dispute auto-escalation) are applied AFTER this in executor.py — this module
only picks the "ideal" action assuming no constraints existed yet.
"""

# root_cause -> ladder of actions by attempt number (1st, 2nd, 3rd try)
ACTION_LADDER = {
    "card_expired":           ["send_payment_link", "send_payment_link", "escalate_human_review"],
    "insufficient_funds":     ["retry_payment", "send_reminder", "escalate_human_review"],
    "auth_friction":          ["send_payment_link", "retry_payment", "escalate_human_review"],
    "bank_decline_other":     ["retry_payment", "send_reminder", "escalate_human_review"],
    "fraud_review_needed":    ["escalate_human_review", "escalate_human_review", "escalate_human_review"],

    "price_sensitivity":      ["send_discount_coupon", "send_reminder", "mark_dead"],
    "missing_payment_method": ["send_payment_link", "send_reminder", "mark_dead"],
    "technical_friction":     ["send_reminder", "send_payment_link", "mark_dead"],
    "low_intent":             ["send_reminder", "mark_dead", "mark_dead"],

    "forgot":                 ["send_reminder", "send_reminder", "escalate_voice_call"],
    "cash_flow_issue":        ["send_reminder", "send_payment_link", "escalate_voice_call"],
    "dispute_pending":        ["escalate_human_review", "escalate_human_review", "escalate_human_review"],
    "invoice_error":          ["escalate_human_review", "escalate_human_review", "escalate_human_review"],
    "awaiting_approval":      ["send_reminder", "escalate_voice_call", "escalate_human_review"],

    "unknown":                ["send_reminder", "escalate_human_review", "mark_dead"],
}

# preferred channel by tier for the "contact" style actions
CHANNEL_BY_TIER = {
    "retail": "sms",
    "smb": "whatsapp",
    "enterprise": "email",
}

# high value B2B nudges escalate to voice faster — reuses your Hinglish voice stack
VOICE_ELIGIBLE_TYPES = {"invoice_overdue"}


def decide_action(txn, root_cause: str) -> tuple[str, str, str]:
    """Returns (action, channel, rationale) — the policy's *proposed* action,
    before guardrails are applied."""
    ladder = ACTION_LADDER.get(root_cause, ACTION_LADDER["unknown"])
    idx = min(txn.attempt_count, len(ladder) - 1)
    action = ladder[idx]

    tier = txn.customer.tier if txn.customer else "retail"
    channel = CHANNEL_BY_TIER.get(tier, "sms")

    if action == "escalate_voice_call" and txn.type not in VOICE_ELIGIBLE_TYPES:
        action = "send_reminder"  # voice reserved for higher-value B2B by default

    if action in ("retry_payment", "escalate_human_review", "mark_dead"):
        if action == "retry_payment":
            channel = "gateway"
        elif action == "escalate_human_review":
            channel = "human_agent"
        else:
            channel = "none"
    elif action == "escalate_voice_call":
        channel = "voice"

    rationale = (
        f"root_cause={root_cause}, attempt={txn.attempt_count + 1}/3, "
        f"tier={tier} -> ladder step '{action}' via {channel}"
    )
    return action, channel, rationale
