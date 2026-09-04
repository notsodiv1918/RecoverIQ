"""
Simulates customer response to an executed action, since this is a synthetic
batch demo (no real payment gateway / SMS provider wired up tonight).
Probabilities are hand-calibrated to be directionally realistic, not exact.
"""
import random

# (action, root_cause) -> (recovery_probability, avg_fraction_of_amount_recovered)
# Calibrated to reflect that a well-targeted intervention (right channel, right
# message, right timing) recovers the clear majority of "soft" failures —
# expired cards, forgotten invoices, cart friction — and meaningfully less on
# genuinely hard cases (disputes, fraud flags, low-intent browsers). Tune these
# to whatever benchmark numbers you want to defend if asked.
BASE_PROB = {
    ("retry_payment", "insufficient_funds"): (0.55, 1.0),
    ("retry_payment", "bank_decline_other"): (0.45, 1.0),
    ("retry_payment", "auth_friction"): (0.40, 1.0),

    ("send_payment_link", "card_expired"): (0.75, 1.0),
    ("send_payment_link", "auth_friction"): (0.65, 1.0),
    ("send_payment_link", "missing_payment_method"): (0.70, 1.0),
    ("send_payment_link", "technical_friction"): (0.60, 1.0),
    ("send_payment_link", "cash_flow_issue"): (0.45, 1.0),

    ("send_discount_coupon", "price_sensitivity"): (0.65, 0.9),

    ("send_reminder", "forgot"): (0.75, 1.0),
    ("send_reminder", "cash_flow_issue"): (0.40, 0.7),
    ("send_reminder", "awaiting_approval"): (0.55, 1.0),
    ("send_reminder", "technical_friction"): (0.50, 1.0),
    ("send_reminder", "low_intent"): (0.15, 1.0),
    ("send_reminder", "bank_decline_other"): (0.35, 1.0),
    ("send_reminder", "insufficient_funds"): (0.35, 1.0),

    ("escalate_voice_call", "forgot"): (0.80, 1.0),
    ("escalate_voice_call", "cash_flow_issue"): (0.55, 0.7),
    ("escalate_voice_call", "awaiting_approval"): (0.70, 1.0),

    ("escalate_human_review", "dispute_pending"): (0.35, 0.6),
    ("escalate_human_review", "invoice_error"): (0.70, 1.0),
    ("escalate_human_review", "fraud_review_needed"): (0.25, 1.0),

    # human review also fires purely on amount threshold (>₹50k) regardless of
    # root cause — a person calling a large account is generally MORE
    # effective than automated channels, so these should recover well:
    ("escalate_human_review", "forgot"): (0.70, 1.0),
    ("escalate_human_review", "cash_flow_issue"): (0.50, 0.75),
    ("escalate_human_review", "awaiting_approval"): (0.65, 1.0),
    ("escalate_human_review", "card_expired"): (0.75, 1.0),
    ("escalate_human_review", "insufficient_funds"): (0.55, 1.0),
    ("escalate_human_review", "missing_payment_method"): (0.65, 1.0),
    ("escalate_human_review", "technical_friction"): (0.60, 1.0),
    ("escalate_human_review", "auth_friction"): (0.60, 1.0),
    ("escalate_human_review", "bank_decline_other"): (0.55, 1.0),
    ("escalate_human_review", "price_sensitivity"): (0.55, 0.9),
}

DEFAULT_PROB = (0.45, 0.85)  # generic fallback if combo not in table


def simulate_outcome(txn, action: str) -> tuple[bool, float]:
    """Returns (recovered: bool, recovered_amount: float)."""
    if action in ("mark_dead", "escalate_human_review") and action == "mark_dead":
        return False, 0.0

    prob, frac = BASE_PROB.get((action, txn.root_cause), DEFAULT_PROB)

    # bigger tickets are slightly harder to recover on first automated touch
    if txn.amount > 50000:
        prob *= 0.85

    recovered = random.random() < prob
    amount = round(txn.amount * frac, 2) if recovered else 0.0
    return recovered, amount
