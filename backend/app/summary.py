"""
Executive AI Summary: a short templated readout generated from real metrics
after each batch run. Deliberately NOT free-text LLM generation -- every
number in the summary is pulled directly from the metrics dict, so it can
never state something the data doesn't support. Same principle as policy.py:
keep anything that must be trustworthy deterministic, use generation only
for phrasing.
"""


def format_inr(n) -> str:
    """Indian digit grouping (lakh/crore: 1,90,98,512), not the Western
    thousands grouping Python's f-string ',' gives you by default. All
    money in this app is Indian Rupees, so it should read that way
    everywhere -- the metric cards used this format already; this brings
    the summary text in line with them."""
    n = int(round(n))
    negative = n < 0
    n = abs(n)
    s = str(n)
    if len(s) <= 3:
        result = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        result = ",".join(parts) + "," + last3
    return ("-" if negative else "") + result


def generate_summary(metrics: dict) -> dict:
    total_at_risk = metrics["total_at_risk"]
    total_recovered = metrics["total_recovered"]
    rate_amount = metrics["recovery_rate_by_amount"]
    rate_count = metrics["recovery_rate_by_count"]
    total_txns = metrics["total_transactions"]
    by_root_cause = metrics.get("by_root_cause", [])
    by_channel = metrics.get("by_channel", [])
    baseline = metrics.get("baseline", {})

    top_cause = max(by_root_cause, key=lambda c: c["count"], default=None)
    best_channel = max(by_channel, key=lambda c: c["recovered_amount"], default=None)

    remaining = total_at_risk - total_recovered

    lines = []
    lines.append(f"\u20b9{format_inr(total_at_risk)} was at risk across {total_txns} transactions.")
    lines.append(
        f"\u20b9{format_inr(total_recovered)} was recovered "
        f"({rate_amount*100:.1f}% of value, {rate_count*100:.1f}% of transactions)."
    )
    if top_cause:
        lines.append(
            f"'{top_cause['root_cause']}' was the most common root cause "
            f"({top_cause['count']} transactions)."
        )
    if best_channel and best_channel["recovered_amount"] > 0:
        lines.append(
            f"'{best_channel['channel']}' was the best-performing recovery channel "
            f"(\u20b9{format_inr(best_channel['recovered_amount'])} recovered across {best_channel['count']} cases)."
        )
    if baseline:
        lift = baseline.get("lift_amount", 0)
        if lift > 0:
            lines.append(
                f"That's \u20b9{format_inr(lift)} more than a flat, unprioritized baseline "
                f"process would have recovered ({baseline.get('lift_pct', 0)*100:.0f}% lift)."
            )
    lines.append(f"\u20b9{format_inr(remaining)} remains under active recovery or escalated to human review.")

    return {
        "headline": f"\u20b9{format_inr(total_recovered)} recovered of \u20b9{format_inr(total_at_risk)} at risk ({rate_amount*100:.1f}%)",
        "bullets": lines,
    }