"""
Budget allocation optimizer.

Given attribution weights and historical spend, finds the
spend allocation that maximises expected conversions subject to:
  • Total budget constraint
  • Per-channel min/max bounds
  • Diminishing returns (log-concave response curves)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .data_generator import CHANNELS, CHANNEL_LABELS, CHANNEL_CPT


def _response_curve(spend: float, alpha: float, beta: float = 0.5) -> float:
    """Diminishing-returns response: alpha * spend^beta."""
    return alpha * (spend ** beta)


def optimize_budget(
    attribution_weights: Dict[str, float],   # channel → fractional credit
    total_budget: float = 100_000,
    min_per_channel: float = 0.01,           # fraction of budget
    max_per_channel: float = 0.50,
    current_spend: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame comparing current vs optimised spend per channel,
    along with expected lift.
    """
    channels = [ch for ch in CHANNELS if attribution_weights.get(ch, 0) > 0]
    n = len(channels)

    # Default current spend: proportional to CPT (higher CPT → more historically spent)
    if current_spend is None:
        cpt_vals = np.array([CHANNEL_CPT[ch] + 1 for ch in channels], dtype=float)
        current_weights = cpt_vals / cpt_vals.sum()
        current_spend = {ch: total_budget * w for ch, w in zip(channels, current_weights)}

    attr = np.array([attribution_weights.get(ch, 0.0) for ch in channels])
    attr = np.maximum(attr, 1e-6)

    # alpha_i ∝ attribution weight (higher credit → higher marginal return)
    alpha = attr / attr.sum()

    def neg_total_response(x: np.ndarray) -> float:
        return -sum(_response_curve(x[i], alpha[i]) for i in range(n))

    def neg_total_response_grad(x: np.ndarray) -> np.ndarray:
        return np.array([-0.5 * alpha[i] * (x[i] ** (-0.5)) for i in range(n)])

    bounds = [
        (total_budget * min_per_channel, total_budget * max_per_channel)
        for _ in channels
    ]
    constraints = {"type": "eq", "fun": lambda x: x.sum() - total_budget}

    # Warm start: attribution-proportional
    x0 = attr / attr.sum() * total_budget
    x0 = np.clip(x0, total_budget * min_per_channel, total_budget * max_per_channel)
    x0 = x0 / x0.sum() * total_budget  # re-normalise

    result = minimize(
        neg_total_response,
        x0,
        jac=neg_total_response_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    opt_spend = result.x if result.success else x0

    rows = []
    for i, ch in enumerate(channels):
        curr = current_spend.get(ch, 0.0)
        opt  = float(opt_spend[i])
        rows.append({
            "channel": ch,
            "channel_label": CHANNEL_LABELS[ch],
            "attribution_weight": attr[i],
            "current_spend": curr,
            "optimised_spend": opt,
            "delta": opt - curr,
            "delta_pct": (opt - curr) / max(curr, 1) * 100,
            "current_response": _response_curve(curr, alpha[i]),
            "optimised_response": _response_curve(opt, alpha[i]),
            "response_lift": _response_curve(opt, alpha[i]) - _response_curve(curr, alpha[i]),
        })

    df = pd.DataFrame(rows).sort_values("optimised_spend", ascending=False)
    return df
