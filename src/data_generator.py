"""
Synthetic customer journey data generator.
Produces realistic multi-touch journeys spanning online & offline channels
with configurable parameters.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

# ── Channel registry ──────────────────────────────────────────────────────────
CHANNELS = [
    "paid_search", "display", "paid_social", "email",
    "organic_search", "direct",            # online
    "tv", "radio", "direct_mail", "agent_visit",  # offline
]

CHANNEL_LABELS = {
    "paid_search":    "Paid Search",
    "display":        "Display",
    "paid_social":    "Paid Social",
    "email":          "Email",
    "organic_search": "Organic Search",
    "direct":         "Direct",
    "tv":             "TV",
    "radio":          "Radio",
    "direct_mail":    "Direct Mail",
    "agent_visit":    "Agent Visit",
}

CHANNEL_TYPE = {ch: ("Online" if ch in
                     ["paid_search","display","paid_social","email","organic_search","direct"]
                     else "Offline")
                for ch in CHANNELS}

# Average cost-per-touchpoint (USD) – used in budget optimizer
CHANNEL_CPT = {
    "paid_search": 42,
    "display": 4,
    "paid_social": 22,
    "email": 1,
    "organic_search": 0,
    "direct": 0,
    "tv": 180,
    "radio": 55,
    "direct_mail": 12,
    "agent_visit": 95,
}

CHANNEL_COLORS = {
    "paid_search":    "#1f77b4",
    "display":        "#aec7e8",
    "paid_social":    "#6baed6",
    "email":          "#2196F3",
    "organic_search": "#26a69a",
    "direct":         "#4db6ac",
    "tv":             "#e6550d",
    "radio":          "#fd8d3c",
    "direct_mail":    "#fdae6b",
    "agent_visit":    "#d62728",
}

# ── Journey generation ────────────────────────────────────────────────────────
# Probability that each channel appears as a touchpoint in a given journey
CHANNEL_APPEARANCE = {
    "paid_search":    0.40,
    "display":        0.35,
    "paid_social":    0.30,
    "email":          0.25,
    "organic_search": 0.30,
    "direct":         0.20,
    "tv":             0.20,
    "radio":          0.15,
    "direct_mail":    0.12,
    "agent_visit":    0.18,
}

# Conversion lift added by each channel (log-odds scale)
CHANNEL_LIFT = {
    "paid_search":    0.55,
    "display":        0.20,
    "paid_social":    0.30,
    "email":          0.45,
    "organic_search": 0.40,
    "direct":         0.35,
    "tv":             0.25,
    "radio":          0.15,
    "direct_mail":    0.28,
    "agent_visit":    1.10,   # strongest offline channel
}

# Interaction boosts (multiplicative on log-odds)
SYNERGIES = {
    frozenset(["tv", "paid_search"]): 0.25,
    frozenset(["email", "agent_visit"]): 0.40,
    frozenset(["direct_mail", "agent_visit"]): 0.30,
    frozenset(["paid_social", "email"]): 0.20,
    frozenset(["display", "paid_search"]): 0.15,
}

# Channel funnel position (used for ordered Shapley & Sankey)
CHANNEL_FUNNEL = {
    "tv":             1,
    "radio":          1,
    "display":        2,
    "paid_social":    2,
    "direct_mail":    3,
    "paid_search":    3,
    "organic_search": 3,
    "email":          4,
    "agent_visit":    5,
    "direct":         5,
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def generate_journeys(
    n_customers: int = 3000,
    seed: int = 42,
    start_date: str = "2024-01-01",
    n_days: int = 90,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Returns
    -------
    touchpoints_df : long-format DataFrame with one row per touchpoint
    journeys       : list of dicts with 'path', 'converted', 'value', 'customer_id'
    """
    rng = np.random.default_rng(seed)
    base_dt = datetime.strptime(start_date, "%Y-%m-%d")

    touchpoints_rows = []
    journeys = []

    for cust_id in range(n_customers):
        # ── decide which channels this customer encounters ──────────────────
        present = {
            ch: rng.random() < CHANNEL_APPEARANCE[ch]
            for ch in CHANNELS
        }
        # guarantee at least 1 channel
        if not any(present.values()):
            ch = rng.choice(CHANNELS)
            present[ch] = True

        active = [ch for ch, v in present.items() if v]

        # ── compute conversion probability ───────────────────────────────────
        log_odds = -2.5  # base ~7.5% conversion
        for ch in active:
            log_odds += CHANNEL_LIFT[ch]
        for pair, boost in SYNERGIES.items():
            if pair.issubset(set(active)):
                log_odds += boost
        conv_prob = _sigmoid(log_odds)
        converted = rng.random() < conv_prob

        # ── build temporal sequence (respect funnel order + noise) ───────────
        journey_start = base_dt + timedelta(days=int(rng.integers(0, n_days - 14)))
        ordered = sorted(active, key=lambda c: CHANNEL_FUNNEL[c] + rng.uniform(-0.3, 0.3))
        n_touches = len(ordered)
        span_days = rng.integers(1, min(30, n_touches * 5 + 1))
        offsets = sorted(rng.uniform(0, span_days, n_touches))

        # ── write touchpoints ────────────────────────────────────────────────
        for i, (ch, off) in enumerate(zip(ordered, offsets)):
            ts = journey_start + timedelta(days=float(off))
            touchpoints_rows.append({
                "customer_id": cust_id,
                "channel": ch,
                "channel_label": CHANNEL_LABELS[ch],
                "channel_type": CHANNEL_TYPE[ch],
                "timestamp": ts,
                "position": i + 1,
                "journey_length": n_touches,
                "converted": converted,
            })

        # ── compute deal value (conversion only) ─────────────────────────────
        value = 0.0
        if converted:
            value = rng.normal(1200, 400)  # avg premium ~$1,200
            value = max(400, value)

        journeys.append({
            "customer_id": cust_id,
            "path": ordered,           # ordered list of channel keys
            "path_labels": [CHANNEL_LABELS[c] for c in ordered],
            "converted": converted,
            "value": value,
            "n_touches": n_touches,
        })

    df = pd.DataFrame(touchpoints_rows)
    return df, journeys


def journey_summary(journeys: List[Dict]) -> pd.DataFrame:
    """Aggregate stats by channel."""
    rows = []
    for j in journeys:
        for ch in set(j["path"]):
            rows.append({"channel": ch, "converted": j["converted"], "value": j["value"]})
    agg = (
        pd.DataFrame(rows)
        .groupby("channel")
        .agg(
            touchpoints=("converted", "count"),
            conversions=("converted", "sum"),
            revenue=("value", "sum"),
        )
        .reset_index()
    )
    agg["conv_rate"] = agg["conversions"] / agg["touchpoints"]
    agg["channel_label"] = agg["channel"].map(CHANNEL_LABELS)
    agg["channel_type"] = agg["channel"].map(CHANNEL_TYPE)
    return agg.sort_values("conversions", ascending=False)


def top_paths(journeys: List[Dict], n: int = 15) -> pd.DataFrame:
    """Most common converting paths."""
    from collections import Counter
    c = Counter(
        " → ".join(j["path_labels"])
        for j in journeys if j["converted"]
    )
    rows = [{"path": p, "count": cnt} for p, cnt in c.most_common(n)]
    return pd.DataFrame(rows)
