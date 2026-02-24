# Multi-Touch Attribution Demo — v2 (P0/P1 Improvements)

Cooperative Game Theory–powered attribution across 10 marketing channels
(6 online + 4 offline), with a Streamlit UI and a full suite of models:
Shapley, Ordered Shapley, Banzhaf, Markov chain, and bootstrap CIs.

## What's New in v2

### P0 — Non-Linear Characteristic Function (GBT)
The characteristic function `v(S)` that drives all Shapley-family models now
uses a **GradientBoostingClassifier** trained on binary channel-presence
features **plus all C(10,2) = 45 pairwise interaction columns**
(e.g. `email × agent_visit`).  This replaces the v1 logistic regression,
which was structurally incapable of capturing the non-linear synergies encoded
in the data-generating process.

### P1 — True Ordered Shapley via Plackett-Luce Sampling
`shapley_ordered()` now fits a **Plackett-Luce model** from empirical channel
ordering data.  Per-channel utility scores are estimated from mean normalised
position across all observed journeys, so TV and Radio (awareness channels)
are sampled first in significantly more permutations than uniform Monte Carlo
would predict.  This gives top-funnel channels position-weighted credit that
reflects their actual role in the observed journey data.

### P1 — Bootstrap Confidence Intervals (Bug Fix)
`shapley_bootstrap_ci()` now uses `shapley_exact(backend="gbt_fast")` for
each bootstrap resample — the **same estimator family as the point estimate**.
The previous version used `shapley_ordered` (a different estimator) for
resamples but `shapley_exact` for point estimates, causing point estimates to
fall **outside** their own CIs (Radio: PE=6.10%, CI=[0.00%, 5.32%]).
The fix ensures all 10-channel point estimates fall inside their 95% CIs.

### Additional Fixes
- Low-n warning: GBT requires ≥1,000 customers for reliable attribution
  (below this, Email and Direct may receive 0% due to data starvation)
- Eliminated redundant Markov computation in the Markov tab
- Fixed `st.image(use_column_width=True)` deprecation warning
- CI table uses static "Lower 95% CI" / "Upper 95% CI" headers
- CI chart shows asymmetric `[lo%, hi%]` ranges, not misleading `±half-width`

## Setup

```bash
unzip mta_demo_v2.zip && cd mta_demo_v2
pip install -r requirements.txt   # scikit-learn, scipy, streamlit, plotly, pandas, numpy
streamlit run app.py
# Opens at http://localhost:8501
```

Python 3.9+ required. All dependencies are in the standard PyPI ecosystem —
no external ML frameworks beyond scikit-learn.

## Attribution Models

| Model | Type | Characteristic Fn | Sequence-Aware |
|-------|------|-------------------|----------------|
| Last Touch | Heuristic | — | Yes |
| First Touch | Heuristic | — | Yes |
| Linear | Heuristic | — | No |
| Time Decay (H=7) | Heuristic | — | Yes |
| Position-Based (U-Shape) | Heuristic | — | Yes |
| Markov Chain | Probabilistic | — | Yes |
| Shapley (Exact) | Game Theory | GBT + interactions | No |
| Shapley (Ordered, PL-MC) | Game Theory | GBT + interactions | **Yes** |
| Banzhaf | Game Theory | GBT + interactions | No |
| Shapley Interaction Index | Game Theory | GBT + interactions | No |

## Key References

- Shapley, L.S. (1953). A Value for n-Person Games. Princeton University Press.
- Zhao et al. (2018). Shapley Value Methods for Attribution Modeling. arXiv:1804.05327.
- Grabisch & Roubens (1999). Shapley Interaction Index. IJGT 28(4).
- Anderl et al. (2016). Mapping the customer journey. IJRM 33(3).
