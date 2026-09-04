import datetime as dt
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
)
from sqlalchemy.orm import relationship
from .database import Base


def now():
    return dt.datetime.utcnow()


class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    phone = Column(String, nullable=False)
    email = Column(String, nullable=False)
    tier = Column(String, default="retail")   # retail / smb / enterprise
    opt_out = Column(Boolean, default=False)  # marketing opt-out
    dnd = Column(Boolean, default=False)      # do-not-disturb (voice/sms)

    transactions = relationship("Transaction", back_populates="customer")


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"))

    type = Column(String, nullable=False)       # payment_failed | checkout_abandoned | invoice_overdue
    amount = Column(Float, nullable=False)
    currency = Column(String, default="INR")
    failure_code = Column(String, nullable=True)
    days_overdue = Column(Integer, default=0)

    status = Column(String, default="new")      # new | in_progress | recovered | dead | escalated | pending_approval
    root_cause = Column(String, nullable=True)
    diagnosis_confidence = Column(Float, nullable=True)
    attempt_count = Column(Integer, default=0)
    recovered_amount = Column(Float, default=0.0)

    created_at = Column(DateTime, default=now)
    last_action_batch = Column(Integer, nullable=True)  # batch counter, not wall-clock

    customer = relationship("Customer", back_populates="transactions")
    actions = relationship("ActionLog", back_populates="transaction")
    audit_events = relationship("AuditEvent", back_populates="transaction")


class ActionLog(Base):
    __tablename__ = "actions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"))

    action_type = Column(String, nullable=False)   # retry_payment | send_reminder | send_discount | ...
    channel = Column(String, nullable=True)         # email | sms | whatsapp | voice | none
    rationale = Column(Text, nullable=True)
    outcome = Column(String, nullable=True)         # recovered | no_response | deferred_quiet_hours | blocked_dnd | pending_approval
    recovered_amount = Column(Float, default=0.0)
    sim_hour = Column(Integer, nullable=True)       # simulated clock hour used for quiet-hour check
    link_url = Column(String, nullable=True)        # set for send_payment_link -- real Razorpay
                                                     # test-mode link if configured, else a
                                                     # clearly-labeled simulated one (see razorpay_client.py)
    link_is_real = Column(Boolean, nullable=True)

    timestamp = Column(DateTime, default=now)

    transaction = relationship("Transaction", back_populates="actions")


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"))

    stage = Column(String, nullable=False)   # detect | diagnose | policy_decision | guardrail | action | outcome
    payload = Column(Text, nullable=True)    # JSON string
    timestamp = Column(DateTime, default=now)

    transaction = relationship("Transaction", back_populates="audit_events")


class Meta(Base):
    """Single-row table holding the global batch counter. Cooldowns are
    measured in 'how many batch runs ago' instead of real wall-clock time,
    so a demo that runs entirely in a few minutes can still show a
    transaction being deferred once and then clearing on the next click."""
    __tablename__ = "meta"

    id = Column(Integer, primary_key=True)
    batch_counter = Column(Integer, default=0)


class PolicyStat(Base):
    """Beta-Bernoulli posterior for (root_cause, action). This is the actual
    'learning' in the system — alpha/beta accumulate real outcomes across the
    whole batch, and policy decisions are made by Thompson Sampling over
    these posteriors (see bandit.py), not a static lookup table."""
    __tablename__ = "policy_stats"

    id = Column(Integer, primary_key=True)
    root_cause = Column(String, nullable=False)
    action = Column(String, nullable=False)
    alpha = Column(Float, default=1.0)  # successes + 1 (uniform prior)
    beta = Column(Float, default=1.0)   # failures + 1
    times_chosen = Column(Integer, default=0)