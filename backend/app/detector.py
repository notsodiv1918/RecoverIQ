"""
Detector: turns a raw transaction into a severity score used for prioritization
and dashboard sorting. Not a gate — every txn still goes through diagnosis.
"""

TYPE_BASE_SEVERITY = {
    "payment_failed": 0.5,
    "checkout_abandoned": 0.3,
    "invoice_overdue": 0.6,
}


def severity_score(txn) -> float:
    score = TYPE_BASE_SEVERITY.get(txn.type, 0.4)

    # amount weight (log-ish scaling, capped)
    if txn.amount > 100000:
        score += 0.3
    elif txn.amount > 20000:
        score += 0.2
    elif txn.amount > 5000:
        score += 0.1

    # overdue days weight
    if txn.days_overdue > 60:
        score += 0.25
    elif txn.days_overdue > 30:
        score += 0.15
    elif txn.days_overdue > 7:
        score += 0.05

    # enterprise customers prioritized slightly
    if txn.customer and txn.customer.tier == "enterprise":
        score += 0.1

    return round(min(score, 1.0), 3)
