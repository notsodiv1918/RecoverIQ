"""
Turns the raw audit_events log (detect/diagnose/policy_decision/guardrail/
action/outcome) into the clean journey stages used for the "recovery graph"
visualization: Failed -> Diagnosed -> Prioritised -> Strategy Selected ->
Guardrail Check -> Executed -> Result.

Kept as a separate transform (not baked into the audit log itself) so the
underlying audit trail stays low-level/granular for compliance purposes,
while the UI-facing journey can be redesigned freely without touching
audit semantics.
"""

from .summary import format_inr

STAGE_LABELS = {
    "detect": "Failed / Detected",
    "diagnose": "Diagnosed",
    "policy_decision": "Strategy Selected",
    "guardrail": "Guardrail Check",
    "action": "Deferred",       # only logged for quiet-hours deferrals today
    "outcome": "Result",
}

OUTCOME_TO_RESULT_LABEL = {
    "recovered": "Success",
    "no_response": "No Response",
    "dead": "Written Off",
    "pending_human_review": "Escalated to Human",
    "deferred_quiet_hours": "Deferred (Quiet Hours)",
    "cooldown": "Deferred (Cooldown)",
}


def build_journey(audit_events: list[dict]) -> list[dict]:
    """audit_events: list of {stage, payload, timestamp} in chronological order."""
    journey = []
    for ev in audit_events:
        stage = ev["stage"]
        payload = ev["payload"] or {}
        label = STAGE_LABELS.get(stage, stage)

        detail = ""
        if stage == "detect":
            detail = f"severity {payload.get('severity', '?')}"
        elif stage == "diagnose":
            detail = f"{payload.get('root_cause', '?')} ({payload.get('confidence', 0)*100:.0f}% confidence)"
        elif stage == "policy_decision":
            detail = f"{payload.get('proposed_action', '?')} via {payload.get('proposed_channel', '?')}"
        elif stage == "guardrail":
            detail = payload.get("note", "")
        elif stage == "action":
            detail = payload.get("outcome", "")
        elif stage == "outcome":
            outcome = payload.get("outcome", "")
            label = OUTCOME_TO_RESULT_LABEL.get(outcome, "Result")
            amt = payload.get("recovered_amount", 0)
            detail = f"\u20b9{format_inr(amt)} recovered" if amt else "no amount recovered"

        journey.append({
            "stage": stage,
            "label": label,
            "detail": detail,
            "timestamp": ev["timestamp"],
        })
    return journey