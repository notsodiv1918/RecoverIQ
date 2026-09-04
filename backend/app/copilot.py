"""
Merchant Copilot -- bounded, not freeform.

Every question maps to one of a FIXED set of intents, each backed by a real
query function against the DB (same data every other endpoint uses -- the
copilot has no separate view of the world). If GROQ_API_KEY is set, an LLM
is used ONLY to pick which of these fixed intents best matches free-text
phrasing (constrained to reply with one known intent id, nothing else) --
it never generates the answer itself, so it can't hallucinate a number.
Without a key, a keyword-matching router does the same routing job with
zero external dependency.

The "conversational" feel comes from robust routing over a small, well-
tested set of queries -- not from an LLM writing prose over revenue numbers.
"""
from sqlalchemy import func
from sqlalchemy.orm import Session
from . import models, config, simulator, bandit
from .detector import severity_score
from .summary import generate_summary
from .runner import run_batch as _run_batch
from .summary import format_inr


def _metrics_snapshot(db: Session) -> dict:
    """Minimal recompute of what generate_summary() needs, kept local to
    avoid importing main.py (would create a circular import)."""
    total_at_risk = db.query(func.sum(models.Transaction.amount)).scalar() or 0
    total_recovered = db.query(func.sum(models.Transaction.recovered_amount)).scalar() or 0
    total_txns = db.query(models.Transaction).count()
    status_counts = dict(
        db.query(models.Transaction.status, func.count(models.Transaction.id))
        .group_by(models.Transaction.status).all()
    )
    by_root_cause = [
        {"root_cause": rc or "undiagnosed", "recovered_amount": r or 0, "count": n}
        for rc, r, n in db.query(
            models.Transaction.root_cause,
            func.sum(models.Transaction.recovered_amount),
            func.count(models.Transaction.id),
        ).group_by(models.Transaction.root_cause).all()
    ]
    by_channel = [
        {"channel": c or "none", "recovered_amount": r or 0, "count": n}
        for c, r, n in db.query(
            models.ActionLog.channel,
            func.sum(models.ActionLog.recovered_amount),
            func.count(models.ActionLog.id),
        ).filter(models.ActionLog.outcome == "recovered").group_by(models.ActionLog.channel).all()
    ]
    recovery_rate_amount = (total_recovered / total_at_risk) if total_at_risk else 0
    recovered_count = status_counts.get("recovered", 0)
    resolved_count = sum(status_counts.get(s, 0) for s in ("recovered", "dead", "escalated"))
    recovery_rate_count = (recovered_count / resolved_count) if resolved_count else 0
    baseline_recovered = total_at_risk * config.BASELINE_RECOVERY_RATE
    lift_amount = total_recovered - baseline_recovered
    lift_pct = (lift_amount / baseline_recovered) if baseline_recovered else 0
    return {
        "total_transactions": total_txns, "total_at_risk": total_at_risk,
        "total_recovered": total_recovered, "recovery_rate_by_amount": recovery_rate_amount,
        "recovery_rate_by_count": recovery_rate_count, "status_counts": status_counts,
        "by_root_cause": by_root_cause, "by_channel": by_channel,
        "baseline": {
            "baseline_rate": config.BASELINE_RECOVERY_RATE, "baseline_recovered": baseline_recovered,
            "ai_recovered": total_recovered, "lift_amount": lift_amount, "lift_pct": lift_pct,
        },
    }


# ---------------- intent handlers (all real DB queries, no LLM-generated numbers) ----------------

def _handle_leakage(db: Session) -> dict:
    rows = (
        db.query(models.Transaction.type, func.sum(models.Transaction.amount), func.count(models.Transaction.id))
        .group_by(models.Transaction.type).all()
    )
    by_type = sorted(
        [{"type": t, "amount": a or 0, "count": n} for t, a, n in rows],
        key=lambda r: -r["amount"],
    )

    cause_rows = (
        db.query(models.Transaction.root_cause, func.sum(models.Transaction.amount), func.count(models.Transaction.id))
        .filter(models.Transaction.status != "recovered")
        .group_by(models.Transaction.root_cause).all()
    )
    by_cause = sorted(
        [{"root_cause": rc or "undiagnosed", "amount": a or 0, "count": n} for rc, a, n in cause_rows],
        key=lambda r: -r["amount"],
    )

    if not by_type or not by_cause:
        return {"answer": "No transactions to analyze yet -- seed and run a batch first.",
                "data": {"by_type": by_type, "by_root_cause_unrecovered": by_cause}}

    top_type, top_cause = by_type[0], by_cause[0]
    answer = (
        f"'{top_type['type']}' is the largest source of at-risk revenue "
        f"(\u20b9{format_inr(top_type['amount'])} across {top_type['count']} transactions). "
        f"Of revenue still unrecovered, '{top_cause['root_cause']}' is the biggest single root cause "
        f"(\u20b9{format_inr(top_cause['amount'])} outstanding across {top_cause['count']} cases)."
    )
    return {"answer": answer, "data": {"by_type": by_type, "by_root_cause_unrecovered": by_cause}}


def _handle_priority(db: Session, limit: int = 5) -> dict:
    txns = (
        db.query(models.Transaction)
        .filter(models.Transaction.status.in_(["new", "in_progress"]))
        .all()
    )
    ranked = sorted(txns, key=lambda t: severity_score(t), reverse=True)[:limit]
    items = [
        {
            "id": t.id, "customer_name": t.customer.name if t.customer else None,
            "type": t.type, "amount": t.amount, "root_cause": t.root_cause,
            "severity": severity_score(t),
        }
        for t in ranked
    ]
    if not items:
        answer = "Nothing open right now -- every transaction has been resolved, escalated, or written off."
    else:
        lines = [
            f"#{it['id']} ({it['customer_name']}, \u20b9{format_inr(it['amount'])}, {it['root_cause'] or 'undiagnosed'})"
            for it in items
        ]
        answer = "Highest-priority open transactions right now: " + "; ".join(lines) + "."
    return {"answer": answer, "data": {"priority_queue": items}}


def _handle_potential_recovery(db: Session) -> dict:
    open_txns = (
        db.query(models.Transaction)
        .filter(models.Transaction.status.in_(["new", "in_progress"]))
        .filter(models.Transaction.root_cause.isnot(None))
        .all()
    )
    total_open_amount = sum(t.amount for t in open_txns)
    stat_cache = {(s.root_cause, s.action): s for s in db.query(models.PolicyStat).all()}

    expected_recovery = 0.0
    for t in open_txns:
        candidates = bandit.CANDIDATE_ACTIONS.get(t.root_cause, bandit.CANDIDATE_ACTIONS["unknown"])
        best_action, best_mean = candidates[0], -1.0
        for a in candidates:
            stat = stat_cache.get((t.root_cause, a))
            mean = (stat.alpha / (stat.alpha + stat.beta)) if stat else 0.5
            if mean > best_mean:
                best_mean, best_action = mean, a
        prob, frac = simulator.BASE_PROB.get((best_action, t.root_cause), simulator.DEFAULT_PROB)
        expected_recovery += t.amount * prob * frac

    rate = (expected_recovery / total_open_amount) if total_open_amount else 0
    answer = (
        f"\u20b9{format_inr(total_open_amount)} is still open across {len(open_txns)} transactions. "
        f"Based on what the system has learned so far about which actions work for each root cause, "
        f"we'd expect to recover roughly \u20b9{format_inr(expected_recovery)} ({rate*100:.0f}%) of that if fully worked through."
    )
    return {"answer": answer, "data": {
        "total_open_amount": round(total_open_amount, 2),
        "open_transaction_count": len(open_txns),
        "expected_recovery": round(expected_recovery, 2),
        "expected_recovery_rate": round(rate, 4),
    }}


def _handle_run_recovery(db: Session) -> dict:
    result = _run_batch(db)
    answer = (
        f"Ran a recovery batch: processed {result['processed']} transactions "
        f"(batch #{result['batch_number']}). {result['remaining_open']} transactions remain open."
    )
    return {"answer": answer, "data": result}


def _handle_summary(db: Session) -> dict:
    s = generate_summary(_metrics_snapshot(db))
    return {"answer": " ".join(s["bullets"]), "data": s}


# ---------------- fixed intent registry ----------------

INTENTS = [
    {
        "id": "leakage_breakdown",
        "keywords": ["why", "fall", "losing", "leak", "biggest source"],
        "examples": ["Why did revenue fall today?", "Where are we losing the most money?"],
        "handler": _handle_leakage,
    },
    {
        "id": "priority_queue",
        "keywords": ["recover first", "priorit", "what should", "start with", "tackle first"],
        "examples": ["What should I recover first?"],
        "handler": _handle_priority,
    },
    {
        "id": "potential_recovery",
        "keywords": ["how much can", "potential", "this week", "could we recover", "projected"],
        "examples": ["How much can we potentially recover this week?"],
        "handler": _handle_potential_recovery,
    },
    {
        "id": "run_recovery",
        "keywords": ["run recovery", "run the recovery", "process eligible", "execute recovery", "work the queue"],
        "examples": ["Run recovery for all eligible transactions."],
        "handler": _handle_run_recovery,
    },
    {
        "id": "executive_summary",
        "keywords": ["summary", "how are we doing", "overview", "status", "recap"],
        "examples": ["Give me a summary."],
        "handler": _handle_summary,
    },
]


def _keyword_route(question: str):
    q = question.lower()
    best_id, best_score = None, 0
    for intent in INTENTS:
        score = sum(1 for kw in intent["keywords"] if kw in q)
        if score > best_score:
            best_score, best_id = score, intent["id"]
    return best_id


def _llm_route(question: str):
    if not config.USE_LLM_DIAGNOSIS:
        return None
    try:
        from groq import Groq
        client = Groq(api_key=config.GROQ_API_KEY)
        ids = [i["id"] for i in INTENTS]
        system = (
            f"Classify the merchant's question into exactly one of these intent ids: {ids}. "
            "Reply with ONLY the id, nothing else. If none fit well, reply 'none'."
        )
        resp = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": question}],
            temperature=0, max_tokens=20,
        )
        raw = resp.choices[0].message.content.strip().strip('"').strip("'")
        return raw if raw in ids else None
    except Exception:
        return None


def ask(db: Session, question: str) -> dict:
    intent_id = _llm_route(question) or _keyword_route(question)

    if not intent_id:
        supported = [i["examples"][0] for i in INTENTS]
        return {
            "intent": None,
            "answer": (
                "I can only answer a fixed set of questions right now, not open-ended ones. "
                "Try one of: " + " / ".join(supported)
            ),
            "data": {},
        }

    intent = next(i for i in INTENTS if i["id"] == intent_id)
    result = intent["handler"](db)
    return {"intent": intent_id, "answer": result["answer"], "data": result["data"]}