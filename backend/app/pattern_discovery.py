"""
Sub-Cluster Recovery-Rate Deviation Analysis (formerly pitched as "New
Failure Pattern Discovery" — renamed to say plainly what it does).

The first version clustered the whole population on raw features and
flagged clusters with mixed root causes. That signal turned out to be
meaningless here: diagnosis is a deterministic function of (type,
failure_code), so clustering on amount/days/tier can never align with root
cause boundaries regardless of whether there's a real pattern — everything
gets flagged, which is a threshold artifact, not a finding.

This version asks a more honest question: WITHIN a single diagnosed root
cause, is there a sub-population that behaves very differently from the
group? e.g. "cash_flow_issue" transactions recover at 45% overall, but
enterprise-tier ones in that bucket recover at only 20% — the current
policy treats all "cash_flow_issue" cases identically, so a sub-cluster
like that is a genuine argument for splitting the strategy further.

KMeans still does the actual clustering (on amount/days_overdue/tier within
each root-cause group); the "pattern" is a statistically flagged deviation
in real recovery rate between a sub-cluster and its parent group, not a
guess about what the pattern means, and not a claim of discovering a new
failure type that diagnosis.py doesn't already know about.
"""
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from .summary import format_inr
from sqlalchemy.orm import Session
from . import models

TIER_ENCODING = {"retail": 0, "smb": 1, "enterprise": 2}

MIN_GROUP_SIZE = 15          # need this many resolved txns for a root cause to be worth analyzing
MIN_SUBCLUSTER_SIZE = 5      # ignore tiny sub-clusters as noise
DEVIATION_THRESHOLD = 0.15   # flag sub-clusters whose recovery rate differs from the group's by >15pts


def _feature_vector(txn) -> list[float]:
    tier = txn.customer.tier if txn.customer else "retail"
    return [
        np.log1p(txn.amount),
        txn.days_overdue,
        txn.attempt_count,
        TIER_ENCODING.get(tier, 0),
    ]


def discover_patterns(db: Session, max_sub_clusters: int = 3) -> dict:
    # only resolved transactions — need a real known outcome to measure a recovery rate
    txns = (
        db.query(models.Transaction)
        .filter(models.Transaction.root_cause.isnot(None))
        .filter(models.Transaction.status.in_(["recovered", "dead", "escalated"]))
        .all()
    )

    by_cause: dict = {}
    for t in txns:
        by_cause.setdefault(t.root_cause, []).append(t)

    findings = []
    for cause, members in by_cause.items():
        if len(members) < MIN_GROUP_SIZE:
            continue

        overall_rate = sum(1 for t in members if t.status == "recovered") / len(members)

        k = min(max_sub_clusters, max(2, len(members) // 10))
        X = np.array([_feature_vector(t) for t in members])
        mean, std = X.mean(axis=0), X.std(axis=0)
        std[std == 0] = 1.0
        X_scaled = (X - mean) / std

        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X_scaled)

        for cid in range(k):
            sub_members = [m for m, lbl in zip(members, labels) if lbl == cid]
            if len(sub_members) < MIN_SUBCLUSTER_SIZE:
                continue

            sub_rate = sum(1 for t in sub_members if t.status == "recovered") / len(sub_members)
            deviation = sub_rate - overall_rate
            if abs(deviation) < DEVIATION_THRESHOLD:
                continue  # behaves like the rest of the group — not a finding

            avg_amount = sum(t.amount for t in sub_members) / len(sub_members)
            avg_days = sum(t.days_overdue for t in sub_members) / len(sub_members)
            tiers = Counter(t.customer.tier if t.customer else "retail" for t in sub_members)
            dominant_tier = tiers.most_common(1)[0][0]
            direction = "underperforming" if deviation < 0 else "outperforming"
            verb = "drop" if direction == "underperforming" else "lift"

            findings.append({
                "root_cause": cause,
                "sub_cluster_size": len(sub_members),
                "group_overall_recovery_rate": round(overall_rate, 3),
                "sub_cluster_recovery_rate": round(sub_rate, 3),
                "deviation_pts": round(deviation * 100, 1),
                "direction": direction,
                "avg_amount": round(avg_amount, 2),
                "avg_days_overdue": round(avg_days, 1),
                "dominant_tier": dominant_tier,
                "sample_transaction_ids": [t.id for t in sub_members[:5]],
                "insight": (
                    f"Within '{cause}' (overall {overall_rate*100:.0f}% recovery), a sub-group of "
                    f"{len(sub_members)} \u2014 mostly {dominant_tier} tier, avg \u20b9{format_inr(avg_amount)} \u2014 "
                    f"recovers at {sub_rate*100:.0f}%, a {abs(deviation)*100:.0f}pt {verb} "
                    f"from the group average. Current policy treats all '{cause}' cases the same; "
                    f"this sub-group may warrant a distinct strategy."
                ),
            })

    findings.sort(key=lambda f: -abs(f["deviation_pts"]))

    return {
        "transactions_analyzed": len(txns),
        "root_causes_analyzed": len(by_cause),
        "root_causes_with_enough_data": sum(1 for m in by_cause.values() if len(m) >= MIN_GROUP_SIZE),
        "patterns_found": len(findings),
        "findings": findings,
    }