import json
from fastapi import FastAPI, Depends, Query, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func

from . import models, config
from .database import init_db, get_db, SessionLocal
from .synthetic_data import generate_customers, generate_transactions
from .executor import process_transaction
from .detector import severity_score
from .summary import generate_summary
from .journey import build_journey
from . import policy, simulator, bandit
from .pattern_discovery import discover_patterns
from . import runner, copilot

app = FastAPI(title="AI Revenue Recovery Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/api/health")
def health():
    return {"status": "ok", "llm_diagnosis_enabled": config.USE_LLM_DIAGNOSIS}


@app.post("/api/seed")
def seed(n_customers: int = 150, n_transactions: int = 300, db: Session = Depends(get_db)):
    customers = generate_customers(db, n_customers)
    txns = generate_transactions(db, customers, n_transactions)
    return {"customers_created": len(customers), "transactions_created": len(txns)}


@app.post("/api/reset")
def reset(db: Session = Depends(get_db)):
    db.query(models.AuditEvent).delete()
    db.query(models.ActionLog).delete()
    db.query(models.Transaction).delete()
    db.query(models.Customer).delete()
    db.query(models.Meta).delete()
    db.query(models.PolicyStat).delete()
    db.commit()
    return {"status": "reset"}


def _next_batch_number(db: Session) -> int:
    return runner.next_batch_number(db)


@app.post("/api/run-batch")
def run_batch(limit: int = 1000, db: Session = Depends(get_db)):
    return runner.run_batch(db, limit=limit)


def _count_open(db: Session) -> int:
    return runner.count_open(db)


@app.get("/api/metrics")
def metrics(db: Session = Depends(get_db)):
    total_txns = db.query(models.Transaction).count()
    total_at_risk = db.query(func.sum(models.Transaction.amount)).scalar() or 0
    total_recovered = db.query(func.sum(models.Transaction.recovered_amount)).scalar() or 0

    status_counts = dict(
        db.query(models.Transaction.status, func.count(models.Transaction.id))
        .group_by(models.Transaction.status)
        .all()
    )

    # recovery by channel
    channel_rows = (
        db.query(
            models.ActionLog.channel,
            func.sum(models.ActionLog.recovered_amount),
            func.count(models.ActionLog.id),
        )
        .filter(models.ActionLog.outcome == "recovered")
        .group_by(models.ActionLog.channel)
        .all()
    )
    by_channel = [
        {"channel": c or "none", "recovered_amount": r or 0, "count": n}
        for c, r, n in channel_rows
    ]

    # recovery by root cause
    cause_rows = (
        db.query(
            models.Transaction.root_cause,
            func.sum(models.Transaction.recovered_amount),
            func.count(models.Transaction.id),
        )
        .group_by(models.Transaction.root_cause)
        .all()
    )
    by_root_cause = [
        {"root_cause": rc or "undiagnosed", "recovered_amount": r or 0, "count": n}
        for rc, r, n in cause_rows
    ]

    # guardrail effect counts (from ActionLog outcomes)
    outcome_rows = (
        db.query(models.ActionLog.outcome, func.count(models.ActionLog.id))
        .group_by(models.ActionLog.outcome)
        .all()
    )
    outcomes = dict(outcome_rows)

    recovery_rate_amount = (total_recovered / total_at_risk) if total_at_risk else 0
    recovered_count = status_counts.get("recovered", 0)
    resolved_count = sum(
        status_counts.get(s, 0) for s in ("recovered", "dead", "escalated")
    )
    recovery_rate_count = (recovered_count / resolved_count) if resolved_count else 0

    # ---- Baseline vs AI ----
    # Baseline = what a flat, unprioritized, no-diagnosis, single-touch dunning
    # process would statistically recover, applied to the SAME at-risk pool.
    baseline_recovered = total_at_risk * config.BASELINE_RECOVERY_RATE
    lift_amount = total_recovered - baseline_recovered
    lift_pct = (lift_amount / baseline_recovered) if baseline_recovered else 0
    baseline = {
        "baseline_rate": config.BASELINE_RECOVERY_RATE,
        "baseline_recovered": round(baseline_recovered, 2),
        "ai_recovered": round(total_recovered, 2),
        "lift_amount": round(lift_amount, 2),
        "lift_pct": round(lift_pct, 4),
    }

    return {
        "total_transactions": total_txns,
        "total_at_risk": round(total_at_risk, 2),
        "total_recovered": round(total_recovered, 2),
        "recovery_rate": round(recovery_rate_amount, 4),
        "recovery_rate_by_amount": round(recovery_rate_amount, 4),
        "recovery_rate_by_count": round(recovery_rate_count, 4),
        "status_counts": status_counts,
        "by_channel": by_channel,
        "by_root_cause": by_root_cause,
        "action_outcomes": outcomes,
        "baseline": baseline,
    }


@app.get("/api/summary")
def summary(db: Session = Depends(get_db)):
    m = metrics(db)
    return generate_summary(m)


@app.get("/api/policy-stats")
def policy_stats(db: Session = Depends(get_db)):
    """The learned Beta-Bernoulli posteriors driving the bandit — this is
    the literal 'learning from previous recoveries' data. n_observations=0
    rows mean that (root_cause, action) pair hasn't been tried yet."""
    rows = (
        db.query(models.PolicyStat)
        .order_by(models.PolicyStat.root_cause, models.PolicyStat.action)
        .all()
    )
    return [
        {
            "root_cause": r.root_cause,
            "action": r.action,
            "alpha": round(r.alpha, 2),
            "beta": round(r.beta, 2),
            "learned_win_rate": round(r.alpha / (r.alpha + r.beta), 3),
            "n_observations": int(r.alpha + r.beta - 2),
            "times_chosen": r.times_chosen,
        }
        for r in rows
    ]


@app.get("/api/policy-comparison")
def policy_comparison(db: Session = Depends(get_db)):
    """Empirical baseline-vs-AI: for every diagnosed transaction, compares
    what the STATIC ladder (policy.py, first-attempt action) would be
    expected to recover vs what the BANDIT currently believes is the best
    action for that root cause (exploiting its learned posterior mean, not
    sampling). Both sides are scored against the same underlying simulator
    probability table — this does not touch or mutate any persisted stats,
    it's a read-only comparison over the current state of learning."""
    txns = db.query(models.Transaction).filter(models.Transaction.root_cause.isnot(None)).all()

    stat_cache = {(s.root_cause, s.action): s for s in db.query(models.PolicyStat).all()}

    static_expected = 0.0
    learned_expected = 0.0

    for t in txns:
        static_ladder = policy.ACTION_LADDER.get(t.root_cause, policy.ACTION_LADDER["unknown"])
        static_action = static_ladder[0]
        s_prob, s_frac = simulator.BASE_PROB.get((static_action, t.root_cause), simulator.DEFAULT_PROB)
        static_expected += t.amount * s_prob * s_frac

        candidates = bandit.CANDIDATE_ACTIONS.get(t.root_cause, bandit.CANDIDATE_ACTIONS["unknown"])
        best_action, best_mean = candidates[0], -1.0
        for a in candidates:
            stat = stat_cache.get((t.root_cause, a))
            mean = (stat.alpha / (stat.alpha + stat.beta)) if stat else 0.5
            if mean > best_mean:
                best_mean, best_action = mean, a
        b_prob, b_frac = simulator.BASE_PROB.get((best_action, t.root_cause), simulator.DEFAULT_PROB)
        learned_expected += t.amount * b_prob * b_frac

    lift_amount = learned_expected - static_expected
    lift_pct = (lift_amount / static_expected) if static_expected else 0

    return {
        "transactions_evaluated": len(txns),
        "static_policy_expected_recovery": round(static_expected, 2),
        "learned_policy_expected_recovery": round(learned_expected, 2),
        "lift_amount": round(lift_amount, 2),
        "lift_pct": round(lift_pct, 4),
    }


@app.get("/api/patterns")
def patterns(max_sub_clusters: int = 3, db: Session = Depends(get_db)):
    """Within each diagnosed root cause, finds sub-populations whose real
    recovery rate deviates meaningfully from the group average — see
    pattern_discovery.py for why this replaced a naive whole-population
    clustering approach."""
    return discover_patterns(db, max_sub_clusters=max_sub_clusters)


class CopilotQuestion(BaseModel):
    question: str


@app.post("/api/copilot")
def copilot_ask(body: CopilotQuestion, db: Session = Depends(get_db)):
    """Bounded Q&A — routes to one of a fixed set of intents (see
    copilot.py), each backed by a real DB query. Never lets an LLM
    generate the numbers in the answer."""
    return copilot.ask(db, body.question)


@app.get("/api/copilot/examples")
def copilot_examples():
    """The fixed set of supported questions — useful for a UI to render
    as suggestion chips instead of a freeform text box."""
    return [{"id": i["id"], "examples": i["examples"]} for i in copilot.INTENTS]


@app.get("/api/priority-queue")
def priority_queue(limit: int = 20, db: Session = Depends(get_db)):
    """Open transactions ranked by severity — 'what should we work on first'."""
    txns = (
        db.query(models.Transaction)
        .filter(models.Transaction.status.in_(["new", "in_progress"]))
        .all()
    )
    ranked = sorted(txns, key=lambda t: severity_score(t), reverse=True)[:limit]
    return [
        {
            "id": t.id,
            "customer_name": t.customer.name if t.customer else None,
            "customer_tier": t.customer.tier if t.customer else None,
            "type": t.type,
            "amount": t.amount,
            "root_cause": t.root_cause,
            "status": t.status,
            "attempt_count": t.attempt_count,
            "severity": severity_score(t),
        }
        for t in ranked
    ]


@app.get("/api/transactions")
def list_transactions(
    status: str | None = None,
    type: str | None = None,
    limit: int = 200,
    db: Session = Depends(get_db),
):
    q = db.query(models.Transaction)
    if status:
        q = q.filter(models.Transaction.status == status)
    if type:
        q = q.filter(models.Transaction.type == type)
    txns = q.order_by(models.Transaction.id.desc()).limit(limit).all()

    return [
        {
            "id": t.id,
            "customer_name": t.customer.name if t.customer else None,
            "customer_tier": t.customer.tier if t.customer else None,
            "type": t.type,
            "amount": t.amount,
            "failure_code": t.failure_code,
            "days_overdue": t.days_overdue,
            "status": t.status,
            "root_cause": t.root_cause,
            "diagnosis_confidence": t.diagnosis_confidence,
            "attempt_count": t.attempt_count,
            "recovered_amount": t.recovered_amount,
            "severity": severity_score(t),
            "created_at": t.created_at,
        }
        for t in txns
    ]


@app.get("/api/transactions/{txn_id}")
def transaction_detail(txn_id: int, db: Session = Depends(get_db)):
    txn = db.query(models.Transaction).filter(models.Transaction.id == txn_id).first()
    if not txn:
        raise HTTPException(status_code=404, detail="not found")

    actions = (
        db.query(models.ActionLog)
        .filter(models.ActionLog.transaction_id == txn_id)
        .order_by(models.ActionLog.timestamp.asc())
        .all()
    )
    audit = (
        db.query(models.AuditEvent)
        .filter(models.AuditEvent.transaction_id == txn_id)
        .order_by(models.AuditEvent.timestamp.asc())
        .all()
    )

    audit_trail = [
        {"stage": e.stage, "payload": json.loads(e.payload) if e.payload else {}, "timestamp": e.timestamp}
        for e in audit
    ]

    return {
        "transaction": {
            "id": txn.id,
            "customer_name": txn.customer.name if txn.customer else None,
            "type": txn.type,
            "amount": txn.amount,
            "failure_code": txn.failure_code,
            "days_overdue": txn.days_overdue,
            "status": txn.status,
            "root_cause": txn.root_cause,
            "diagnosis_confidence": txn.diagnosis_confidence,
            "attempt_count": txn.attempt_count,
            "recovered_amount": txn.recovered_amount,
            "severity": severity_score(txn),
        },
        "actions": [
            {
                "action_type": a.action_type, "channel": a.channel, "rationale": a.rationale,
                "outcome": a.outcome, "recovered_amount": a.recovered_amount,
                "sim_hour": a.sim_hour, "timestamp": a.timestamp,
            }
            for a in actions
        ],
        "audit_trail": audit_trail,
        "journey": build_journey(audit_trail),
    }


@app.get("/api/audit")
def audit_log(transaction_id: int | None = None, limit: int = 300, db: Session = Depends(get_db)):
    q = db.query(models.AuditEvent)
    if transaction_id:
        q = q.filter(models.AuditEvent.transaction_id == transaction_id)
    events = q.order_by(models.AuditEvent.id.desc()).limit(limit).all()
    return [
        {
            "id": e.id, "transaction_id": e.transaction_id, "stage": e.stage,
            "payload": json.loads(e.payload) if e.payload else {}, "timestamp": e.timestamp,
        }
        for e in events
    ]