"""
Diagnosis stage: given a transaction, determine root_cause + confidence.

Deterministic rule-based mapping is the default and is what the whole pipeline
is validated against (works with zero external dependencies / API keys).

If GROQ_API_KEY is set, we additionally ask an LLM to confirm/refine the
root cause with a rationale — but the LLM output is constrained to the same
enum via function-calling-style structured output, and if it fails/errors we
silently fall back to the rule-based result. The pipeline never blocks on
the LLM being unavailable.
"""
import json
from . import config

# ---- deterministic fallback: (type, failure_code) -> (root_cause, confidence) ----
RULE_MAP = {
    ("payment_failed", "card_expired"): ("card_expired", 0.95),
    ("payment_failed", "insufficient_funds"): ("insufficient_funds", 0.90),
    ("payment_failed", "three_ds_failure"): ("auth_friction", 0.85),
    ("payment_failed", "bank_decline_generic"): ("bank_decline_other", 0.55),
    ("payment_failed", "fraud_flag_soft"): ("fraud_review_needed", 0.60),

    ("checkout_abandoned", "price_friction"): ("price_sensitivity", 0.80),
    ("checkout_abandoned", "payment_method_missing"): ("missing_payment_method", 0.85),
    ("checkout_abandoned", "session_timeout"): ("technical_friction", 0.75),
    ("checkout_abandoned", "shipping_cost_shock"): ("price_sensitivity", 0.70),
    ("checkout_abandoned", "just_browsing"): ("low_intent", 0.50),

    ("invoice_overdue", "forgot"): ("forgot", 0.80),
    ("invoice_overdue", "cash_flow_issue"): ("cash_flow_issue", 0.75),
    ("invoice_overdue", "dispute_pending"): ("dispute_pending", 0.95),
    ("invoice_overdue", "invoice_error"): ("invoice_error", 0.85),
    ("invoice_overdue", "awaiting_internal_approval"): ("awaiting_approval", 0.70),
}


def _rule_based(txn) -> tuple[str, float, str]:
    key = (txn.type, txn.failure_code)
    root_cause, confidence = RULE_MAP.get(key, ("unknown", 0.30))
    rationale = f"Rule-based mapping for type={txn.type}, failure_code={txn.failure_code}"
    return root_cause, confidence, rationale


def _llm_diagnose(txn) -> tuple[str, float, str] | None:
    if not config.USE_LLM_DIAGNOSIS:
        return None
    try:
        from groq import Groq
        client = Groq(api_key=config.GROQ_API_KEY)

        allowed_causes = sorted({v[0] for v in RULE_MAP.values()} | {"unknown"})

        system = (
            "You are a revenue-recovery diagnosis engine. Given a transaction, "
            "output ONLY a JSON object with keys: root_cause, confidence, rationale. "
            f"root_cause MUST be one of: {allowed_causes}. "
            "confidence is a float 0-1. rationale is one short sentence. "
            "No markdown, no preamble, JSON only."
        )
        user = json.dumps({
            "type": txn.type,
            "failure_code": txn.failure_code,
            "amount": txn.amount,
            "days_overdue": txn.days_overdue,
            "customer_tier": txn.customer.tier if txn.customer else "retail",
            "attempt_count": txn.attempt_count,
        })

        resp = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)

        root_cause = data.get("root_cause", "unknown")
        if root_cause not in allowed_causes:
            root_cause = "unknown"
        confidence = float(data.get("confidence", 0.5))
        rationale = str(data.get("rationale", "LLM diagnosis"))
        return root_cause, confidence, f"[LLM] {rationale}"
    except Exception as e:
        return None  # silent fallback — pipeline must never break on LLM failure


def diagnose(txn) -> tuple[str, float, str]:
    llm_result = _llm_diagnose(txn)
    if llm_result is not None:
        return llm_result
    return _rule_based(txn)
