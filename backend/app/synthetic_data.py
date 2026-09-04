import random
from faker import Faker
from sqlalchemy.orm import Session
from . import models

fake = Faker("en_IN")

TIERS = ["retail", "retail", "retail", "smb", "smb", "enterprise"]

FAILURE_CODES = {
    "payment_failed": [
        "card_expired", "insufficient_funds", "three_ds_failure",
        "bank_decline_generic", "fraud_flag_soft",
    ],
    "checkout_abandoned": [
        "price_friction", "payment_method_missing", "session_timeout",
        "shipping_cost_shock", "just_browsing",
    ],
    "invoice_overdue": [
        "forgot", "cash_flow_issue", "dispute_pending",
        "invoice_error", "awaiting_internal_approval",
    ],
}

TYPE_WEIGHTS = [("payment_failed", 0.40), ("checkout_abandoned", 0.35), ("invoice_overdue", 0.25)]


def _weighted_type():
    r = random.random()
    acc = 0
    for t, w in TYPE_WEIGHTS:
        acc += w
        if r <= acc:
            return t
    return TYPE_WEIGHTS[-1][0]


def _amount_for(txn_type, tier):
    if txn_type == "checkout_abandoned":
        base = random.uniform(300, 6000)
    elif txn_type == "payment_failed":
        base = random.uniform(500, 15000)
    else:  # invoice_overdue -> B2B, bigger tickets
        base = random.uniform(5000, 200000)

    tier_mult = {"retail": 1.0, "smb": 2.5, "enterprise": 6.0}[tier]
    return round(base * (tier_mult if txn_type == "invoice_overdue" else 1.0), 2)


def generate_customers(db: Session, n: int):
    customers = []
    for _ in range(n):
        c = models.Customer(
            name=fake.name(),
            phone=fake.phone_number()[:15],
            email=fake.email(),
            tier=random.choice(TIERS),
            opt_out=random.random() < 0.07,
            dnd=random.random() < 0.10,
        )
        db.add(c)
        customers.append(c)
    db.commit()
    for c in customers:
        db.refresh(c)
    return customers


def generate_transactions(db: Session, customers, n: int):
    txns = []
    for _ in range(n):
        cust = random.choice(customers)
        t_type = _weighted_type()
        failure_code = random.choice(FAILURE_CODES[t_type])
        amount = _amount_for(t_type, cust.tier)
        days_overdue = random.randint(1, 90) if t_type == "invoice_overdue" else 0

        txn = models.Transaction(
            customer_id=cust.id,
            type=t_type,
            amount=amount,
            failure_code=failure_code,
            days_overdue=days_overdue,
            status="new",
        )
        db.add(txn)
        txns.append(txn)
    db.commit()
    for t in txns:
        db.refresh(t)
    return txns
