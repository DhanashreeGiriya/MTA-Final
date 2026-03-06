"""
Multi-Touch Attribution Demo — Streamlit Application
=====================================================
Cooperative Game Theory meets Marketing Analytics

Tabs
----
📊  Overview          — KPIs, channel volume, sample journeys
🎯  Model Comparison  — Side-by-side heatmap of all attribution models
🔬  Shapley Deep Dive — Exact Shapley, Ordered Shapley, Banzhaf + waterfall
🔗  Channel Synergies — Shapley Interaction Index pairwise heatmap
🛤️  Journey Explorer  — Sankey flow & top converting paths
💰  Budget Optimizer  — Constrained budget reallocation using Shapley weights
📈  Markov Analysis   — Transition matrix + removal-effect attribution
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src import (
    generate_journeys, journey_summary, top_paths,
    CHANNELS, CHANNEL_LABELS,
    run_all_models, shapley_exact, shapley_ordered,
    banzhaf, shapley_interaction_index, markov_chain,
    shapley_bootstrap_ci,
    optimize_budget,
    attribution_comparison, shapley_waterfall, model_radar,
    interaction_heatmap, journey_sankey, budget_waterfall,
    budget_delta_chart, markov_transition_heatmap,
    channel_funnel_bar, conversion_rate_bar,
    shapley_ci_chart,
)
from src.data_generator import CHANNEL_COLORS, CHANNEL_TYPE, CHANNEL_CPT
from src.mmm_data_generator import generate_mmm_data, mmm_summary_stats, ADSTOCK_DECAY, CHANNEL_TRUE_CONTRIBUTION
from src.hybrid_attribution import (
    blend_mta_mmm, compute_unified_metrics,
    compare_mta_vs_mmm_vs_hybrid, offline_credit_recovery,
    DEFAULT_ALPHA,
)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MTA Demo — Shapley Attribution",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px; padding: 1rem 1.2rem;
        border-left: 4px solid #9b59b6;
        margin-bottom: 0.5rem;
    }
    .metric-card h3 { margin: 0; font-size: 0.8rem; color: #6c757d; }
    .metric-card h2 { margin: 0.2rem 0 0; font-size: 1.6rem; color: #212529; }
    .model-tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }
    .tag-heuristic { background: #ffeaa7; color: #6c5c00; }
    .tag-markov    { background: #dfe6fd; color: #1a3a8f; }
    .tag-shapley   { background: #e8d5f7; color: #5a1e8c; }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px; border-radius: 8px 8px 0 0;
        font-weight: 600; font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/MTA%20Demo-Shapley%20Values-9b59b6?style=for-the-badge",
             width='stretch')
    st.markdown("### ⚙️ Data Configuration")
    n_customers = st.slider("Number of customers", 500, 10_000, 3000, step=500)
    seed = st.number_input("Random seed", 0, 999, 42)

    st.markdown("---")
    st.markdown("### 🧮 Model Settings")

    run_ordered = st.checkbox("Ordered Shapley (Zhao 2018)", value=True)
    run_banzhaf = st.checkbox("Banzhaf Index", value=True)
    run_markov  = st.checkbox("Markov Chain", value=True)

    ordered_samples = st.slider("MC samples (Ordered Shapley)", 200, 3000, 1000, step=200,
                                 help="More samples → more accurate but slower")

    st.markdown("---")
    st.markdown("### 📊 Bootstrap Confidence Intervals")
    run_ci = st.checkbox(
        "Compute Shapley CIs",
        value=False,
        help="Resample journeys 50× to produce 95% bootstrap CIs. Adds ~18 s.",
    )
    n_bootstrap = st.slider(
        "Bootstrap resamples",
        min_value=50, max_value=300, value=50, step=50,
        disabled=not run_ci,
        help="More resamples → narrower, more reliable CIs",
    )

    st.markdown("---")
    st.markdown("### 💰 Budget Optimizer")
    total_budget = st.number_input("Total Budget ($)", 10_000, 1_000_000, 100_000, step=10_000)
    min_alloc = st.slider("Min allocation per channel (%)", 0, 20, 2) / 100
    max_alloc = st.slider("Max allocation per channel (%)", 20, 80, 50) / 100

    st.markdown("---")
    st.markdown("""
    **Models in this demo**

    <span class='model-tag tag-heuristic'>Last Touch</span>
    <span class='model-tag tag-heuristic'>First Touch</span>
    <span class='model-tag tag-heuristic'>Linear</span>
    <span class='model-tag tag-heuristic'>Time Decay</span>
    <span class='model-tag tag-heuristic'>Position-Based</span>
    <span class='model-tag tag-markov'>Markov Chain</span>
    <span class='model-tag tag-shapley'>Shapley</span>
    <span class='model-tag tag-shapley'>Ordered Shapley</span>
    <span class='model-tag tag-shapley'>Banzhaf</span>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Built with Streamlit · Cooperative Game Theory · GBT + Scikit-learn")


# ── Data generation & caching ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(n: int, s: int):
    return generate_journeys(n_customers=n, seed=s)

@st.cache_data(show_spinner=False)
def load_models(journeys_hash, _journeys, run_ord, run_bz, run_mkv, ord_samples):
    return run_all_models(
        _journeys,
        run_shapley=True,
        run_ordered=run_ord,
        run_banzhaf=run_bz,
        run_markov=run_mkv,
        ordered_n_samples=ord_samples,   # was not threaded through before — now fixed
    )

@st.cache_data(show_spinner=False)
def load_interactions(journeys_hash, _journeys):
    return shapley_interaction_index(_journeys)   # uses GBT CF by default

@st.cache_data(show_spinner=False)
def load_markov(journeys_hash, _journeys):
    return markov_chain(_journeys)

@st.cache_data(show_spinner=False)
def load_bootstrap_ci(journeys_hash, _journeys, n_boot):
    """Bootstrap CIs — cached separately so they don't block the main spinner."""
    return shapley_bootstrap_ci(_journeys, n_bootstrap=n_boot, n_mc_per_boot=300, seed=42)


@st.cache_data(show_spinner=False)
def load_mmm_data(n_weeks: int = 104, seed: int = 42):
    return generate_mmm_data(n_weeks=n_weeks, seed=seed)


with st.spinner("🔄 Generating synthetic journeys & running attribution models…"):
    df_tp, journeys = load_data(n_customers, seed)
    journeys_hash = hash((n_customers, seed))

    summary_df = journey_summary(journeys)
    paths_df   = top_paths(journeys, n=20)

    attr_df = load_models(journeys_hash, journeys, run_ordered, run_banzhaf,
                          run_markov, ordered_samples)

# ── Data quality warning ───────────────────────────────────────────────────────
if n_customers < 1000:
    st.warning(
        f"⚠️ **Low customer count ({n_customers:,}):** The GBT characteristic function "
        "requires enough converting journeys (~200+) to reliably estimate coalition values. "
        "Below ~1,000 customers, channels with moderate appearance probability (Email, Direct) "
        "may receive **0% Shapley credit** due to data starvation — not because they are "
        "truly inert. Increase to **≥ 1,000 customers** for reliable attribution results."
    )


# ── Hero Banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="
background: linear-gradient(90deg,#2c3e50,#4b6cb7);
padding:20px;
border-radius:10px;
margin-bottom:20px;
color:white;
">
<h2 style="margin:0;">
Marketing Attribution Intelligence Platform
</h2>
<p style="margin:5px 0 0 0;font-size:16px;">
Unified measurement across digital and offline marketing channels
</p>
<p style="margin:5px 0 0 0;font-size:13px;opacity:0.9;">
Shapley Attribution • Markov Modeling • Marketing Mix Modeling • Budget Optimization
</p>
</div>
""", unsafe_allow_html=True)

# ── Key Insight Callout ───────────────────────────────────────────────────────
st.markdown("""
<div style="
background:#f8f9fa;
border-left:5px solid #9b59b6;
padding:15px;
border-radius:6px;
margin-bottom:20px;
">
<b>Key Insight</b><br>
Traditional last-touch attribution significantly over-credits lower funnel
channels like Paid Search while under-crediting upper funnel channels such as
TV, Display, and Direct Mail. Game-theoretic attribution corrects this bias.
</div>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎯 Marketing Attribution Intelligence")
st.markdown(
    "**Cooperative Game Theory–powered attribution** — Shapley values, Banzhaf index, "
    "Ordered Shapley (Zhao 2018), and Markov chain removal effects, compared against "
    "classic heuristic baselines across **10 channels** (6 online + 4 offline)."
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_compare, tab_shapley, tab_synergy, tab_journey, tab_budget, tab_markov, tab_mmm = st.tabs([
    "📊 Overview",
    "🎯 Model Comparison",
    "🔬 Shapley Deep Dive",
    "🔗 Channel Synergies",
    "🛤️ Journey Explorer",
    "💰 Budget Optimizer",
    "📈 Markov Analysis",
    "🔀 MMM + MTA Hybrid",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    total_conv = sum(1 for j in journeys if j["converted"])
    total_rev  = sum(j["value"] for j in journeys)
    conv_rate  = total_conv / len(journeys) * 100
    avg_touches = np.mean([j["n_touches"] for j in journeys])
    avg_order_val = total_rev / max(total_conv, 1)

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, "Customers Simulated", f"{len(journeys):,}",    "#3498db"),
        (c2, "Total Conversions",   f"{total_conv:,}",       "#27ae60"),
        (c3, "Conversion Rate",     f"{conv_rate:.1f}%",     "#9b59b6"),
        (c4, "Total Revenue",       f"${total_rev:,.0f}",    "#e67e22"),
        (c5, "Avg Journey Length",  f"{avg_touches:.1f} touches","#e74c3c"),
    ]
    for col, label, value, color in kpis:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-color:{color}">
                <h3>{label}</h3>
                <h2 style="color:{color}">{value}</h2>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(channel_funnel_bar(summary_df), width='stretch')
    with col_right:
        st.plotly_chart(conversion_rate_bar(summary_df), width='stretch')

    st.markdown("---")
    st.subheader("🧾 Channel Summary")
    disp = summary_df[["channel_label","channel_type","touchpoints","conversions","conv_rate","revenue"]].copy()
    disp.columns = ["Channel","Type","Touchpoints","Conversions","Conv. Rate","Revenue ($)"]
    disp["Conv. Rate"] = (disp["Conv. Rate"] * 100).round(1).astype(str) + "%"
    disp["Revenue ($)"] = disp["Revenue ($)"].apply(lambda x: f"${x:,.0f}")
    st.dataframe(disp, width='stretch', hide_index=True)

    st.markdown("---")
    st.subheader("📋 Sample Touchpoint Log")
    sample = df_tp.sample(min(50, len(df_tp)), random_state=0).sort_values(["customer_id","timestamp"])
    st.dataframe(sample[["customer_id","channel_label","channel_type",
                          "timestamp","position","journey_length","converted"]]
                 .rename(columns={
                     "customer_id":"Customer","channel_label":"Channel",
                     "channel_type":"Type","timestamp":"Timestamp",
                     "position":"Position","journey_length":"Path Length",
                     "converted":"Converted"
                 }),
                 width='stretch', hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("Attribution Credit Across All Models")
    st.markdown(
        "Each bar shows the **fraction of conversion credit** assigned to a channel "
        "by each model. Heuristics (yellow/green) are deterministic; "
        "Markov (blue) uses removal effects; Shapley/Banzhaf (purple) use cooperative game theory."
    )

    available_models = attr_df.columns.tolist()
    selected = st.multiselect(
        "Select models to compare",
        available_models,
        default=available_models,
    )
    if not selected:
        st.warning("Please select at least one model.")
    else:
        st.plotly_chart(attribution_comparison(attr_df, selected), width='stretch')

        st.markdown("---")
        st.subheader("📊 Attribution Matrix (% credit)")
        pct_df = (attr_df[selected] * 100).round(2)
        st.dataframe(
            pct_df.style.background_gradient(cmap="Purples", axis=None),
            width='stretch',
        )

        st.markdown("---")
        st.subheader("🔍 Channel-Level Model Sensitivity")
        ch_label = st.selectbox("Select channel", attr_df.index.tolist())
        st.plotly_chart(model_radar(attr_df[selected], ch_label), width='stretch')

        # Key insight callout
        shapley_col = "Shapley"
        lt_col = "Last Touch"
        if shapley_col in pct_df.columns and lt_col in pct_df.columns:
            delta = pct_df[shapley_col] - pct_df[lt_col]
            biggest_gain = delta.idxmax()
            biggest_loss = delta.idxmin()
            st.info(
                f"💡 **Key Insight:** Shapley values vs. Last Touch — "
                f"**{biggest_gain}** gains **{delta[biggest_gain]:+.1f}pp** credit "
                f"(under-credited by Last Touch), while **{biggest_loss}** loses "
                f"**{delta[biggest_loss]:+.1f}pp** (over-credited by Last Touch)."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SHAPLEY DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_shapley:
    st.subheader("🔬 Cooperative Game Theory Attribution")
    st.markdown("""
    **Shapley values** (Lloyd Shapley, 1953) are the unique attribution satisfying four axioms:
    *Efficiency* (credit sums to total), *Symmetry* (equal channels get equal credit),
    *Dummy* (non-contributing channels get zero), and *Additivity* (credit is additive across games).

    The **Ordered Shapley** (Zhao et al., 2018) additionally respects the *temporal order*
    of touchpoints — earlier channels get position-weighted marginal credit.
    The **Banzhaf index** gives equal weight to all coalition sizes (vs. Shapley's size-weighted averaging).
    """)

    col1, col2 = st.columns([2, 1])

    with col1:
        if "Shapley" in attr_df.columns:
            sh_vals = {
                ch: float(attr_df.loc[CHANNEL_LABELS[ch], "Shapley"])
                for ch in CHANNELS if CHANNEL_LABELS[ch] in attr_df.index
            }
            st.plotly_chart(shapley_waterfall(sh_vals), width='stretch')
        else:
            st.info("Shapley values not computed.")

    with col2:
        st.markdown("### Shapley vs Baselines")
        compare_models = ["Last Touch", "Linear", "Shapley"]
        if "Shapley (Ordered)" in attr_df.columns:
            compare_models.append("Shapley (Ordered)")
        if "Banzhaf" in attr_df.columns:
            compare_models.append("Banzhaf")

        available = [m for m in compare_models if m in attr_df.columns]
        sub_df = (attr_df[available] * 100).round(1)
        sub_df.columns = [m.replace(" (Ordered)", "\n(Ordered)") for m in sub_df.columns]
        st.dataframe(sub_df.style.background_gradient(cmap="Purples"), width='stretch')

    st.markdown("---")
    st.subheader("📐 Mathematical Formulation")
    with st.expander("Click to view Shapley formula & implementation notes"):
        st.latex(r"""
        \phi_i(v) = \sum_{S \subseteq N \setminus \{i\}}
        \frac{|S|!\,(|N|-|S|-1)!}{|N|!}
        \Big[ v(S \cup \{i\}) - v(S) \Big]
        """)
        st.markdown("""
        Where:
        - $N$ = set of all channels (10 in this demo)
        - $S$ = coalition (subset of channels not including $i$)
        - $v(S)$ = **characteristic function** — estimated via a **Gradient Boosted Tree (GBT)**
          trained on binary channel-presence features **plus all C(10,2) = 45 pairwise
          interaction columns** (e.g. `email × agent_visit`).
          GBT captures non-linear synergies that logistic regression's additive
          log-odds structure cannot represent.
        - The formula averages the **marginal contribution** of channel $i$
          across all $2^{{|N|-1}}$ possible coalitions it could join.
          One GBT is trained once and shared across Shapley, Ordered Shapley, and Banzhaf.

        **Ordered Shapley** ({} samples): permutations are sampled from a
        **Plackett-Luce model** fitted on empirical channel funnel positions.
        Per-channel utility scores are estimated from mean normalised position
        across all observed journeys — channels that appear early (TV, Radio)
        get higher utility and are sampled first more often. Top-funnel channels
        earn more position-weighted credit relative to standard Shapley.

        **Banzhaf**: same GBT v(S) but with uniform coalition weight $\\frac{{1}}{{2^{{n-1}}}}$
        instead of Shapley's size-adjusted factorial weights.
        """.format(ordered_samples))

    if "Shapley" in attr_df.columns and "Shapley (Ordered)" in attr_df.columns:
        st.markdown("---")
        st.subheader("Shapley vs Ordered Shapley — Position Effect")
        diff_df = ((attr_df["Shapley (Ordered)"] - attr_df["Shapley"]) * 100).round(2)
        diff_df = diff_df.reset_index()
        diff_df.columns = ["Channel", "Delta (pp)"]
        colors = ["#27ae60" if d > 0 else "#e74c3c" for d in diff_df["Delta (pp)"]]
        fig_diff = go.Figure(go.Bar(
            x=diff_df["Channel"],
            y=diff_df["Delta (pp)"],
            marker_color=colors,
            text=diff_df["Delta (pp)"].apply(lambda x: f"{x:+.2f}pp"),
            textposition="outside",
        ))
        fig_diff.add_hline(y=0, line_color="#333", line_width=1)
        fig_diff.update_layout(
            title="Credit Shift: Ordered Shapley minus Standard Shapley",
            yaxis_title="Percentage Point Difference",
            height=380, template="plotly_white",
        )
        st.plotly_chart(fig_diff, width='stretch')
        st.caption(
            "Positive = channel gains credit when position/order is accounted for "
            "(top-funnel channels are sampled earlier under Plackett-Luce weighting); "
            "Negative = channel is over-credited when order is ignored."
        )

    st.markdown("---")
    st.subheader("📉 Bootstrap Confidence Intervals")
    if not run_ci:
        st.info(
            "Enable **Compute Shapley CIs** in the sidebar to generate 95% bootstrap "
            "confidence intervals.  Each bar shows the GBT point estimate; whiskers "
            "show the percentile interval across journey resamples."
        )
    else:
        with st.spinner(f"Computing bootstrap CIs ({n_bootstrap} resamples, ~18 s)…"):
            ci_df = load_bootstrap_ci(journeys_hash, journeys, n_bootstrap)

        st.plotly_chart(shapley_ci_chart(ci_df), width='stretch')

        n_valid = int(ci_df["n_valid_boots"].iloc[0])
        if n_valid < n_bootstrap:
            st.warning(
                f"⚠️ {n_bootstrap - n_valid} of {n_bootstrap} bootstrap resamples were "
                "skipped due to degenerate labels (all-converted or all-not-converted). "
                "CIs are based on the remaining resamples."
            )

        st.markdown("---")
        st.subheader("CI Detail Table")
        ci_disp = ci_df[[
            "channel_label", "point_estimate", "lower_ci", "upper_ci",
            "ci_width", "std_error"
        ]].copy()
        for col in ["point_estimate", "lower_ci", "upper_ci", "ci_width", "std_error"]:
            ci_disp[col] = (ci_disp[col] * 100).round(2).astype(str) + "%"
        ci_disp.columns = [
            "Channel", "Point Estimate (GBT-full)",
            "Lower 95% CI", "Upper 95% CI",
            "CI Width", "Std Error"
        ]
        st.dataframe(ci_disp, width='stretch', hide_index=True)
        st.caption(
            f"Point estimates: exact Shapley with GBT (150 trees) on full dataset. "
            f"CIs: {n_valid} valid bootstrap resamples using exact Shapley with "
            f"GBT-fast (50 trees) — same estimator family as the point estimate, "
            f"ensuring the CI correctly quantifies data-sampling uncertainty. "
            f"Corr(GBT-fast, GBT-full Shapley) ≈ 0.96."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CHANNEL SYNERGIES
# ═══════════════════════════════════════════════════════════════════════════════
with tab_synergy:
    st.subheader("🔗 Pairwise Channel Interaction Index")
    st.markdown("""
    The **Shapley Interaction Index** (Grabisch & Roubens, 1999) measures whether
    two channels are *synergistic* (work better together than separately) or
    *substitutable* (overlap in the journeys they drive).

    **Formula:**
    """)
    st.latex(r"""
    \phi_{ij}(v) = \sum_{S \subseteq N \setminus \{i,j\}}
    \frac{|S|!\,(|N|-|S|-2)!}{(|N|-1)!}
    \Big[v(S\cup\{i,j\}) - v(S\cup\{i\}) - v(S\cup\{j\}) + v(S)\Big]
    """)
    st.markdown("**Red** = synergy (channels reinforce each other) · **Blue** = substitution (channels overlap)")

    with st.spinner("Computing Shapley Interaction Index…"):
        int_df = load_interactions(journeys_hash, journeys)

    st.plotly_chart(interaction_heatmap(int_df), width='stretch')

    st.markdown("---")
    st.subheader("Top Synergies & Substitutions")
    rows = []
    for i in int_df.index:
        for j in int_df.columns:
            if i < j:
                rows.append({"Channel A": i, "Channel B": j, "Index": round(int_df.loc[i, j], 5)})
    synergy_df = pd.DataFrame(rows).sort_values("Index", ascending=False)

    col_s, col_sub = st.columns(2)
    with col_s:
        st.markdown("**Top 5 Synergistic Pairs** 🤝")
        top5 = synergy_df.head(5).copy()
        top5["Relationship"] = "Synergy 🟢"
        st.dataframe(top5, width='stretch', hide_index=True)
    with col_sub:
        st.markdown("**Top 5 Substitutable Pairs** ↔️")
        bot5 = synergy_df.tail(5).sort_values("Index").copy()
        bot5["Relationship"] = "Substitution 🔴"
        st.dataframe(bot5, width='stretch', hide_index=True)

    st.info("💡 Use synergistic pairs to inform **media mix** decisions — channels that amplify "
            "each other's impact should be run concurrently, not traded off against each other.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — JOURNEY EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_journey:
    st.subheader("🛤️ Customer Journey Flow")

    sankey_n = st.slider("Number of converting journeys to visualise", 50, 500, 200, step=50)
    st.plotly_chart(journey_sankey(journeys, top_n=sankey_n), width='stretch')

    st.markdown("---")
    col_paths, col_stats = st.columns([3, 2])

    with col_paths:
        st.subheader("🏆 Top Converting Paths")
        st.dataframe(paths_df, width='stretch', hide_index=True)

    with col_stats:
        st.subheader("Journey Length Distribution")
        lengths = [j["n_touches"] for j in journeys]
        conv_lengths = [j["n_touches"] for j in journeys if j["converted"]]
        nonconv_lengths = [j["n_touches"] for j in journeys if not j["converted"]]

        fig_len = go.Figure()
        fig_len.add_trace(go.Histogram(x=conv_lengths, name="Converted",
                                       marker_color="#27ae60", opacity=0.7,
                                       xbins=dict(start=1, end=12, size=1)))
        fig_len.add_trace(go.Histogram(x=nonconv_lengths, name="Not Converted",
                                       marker_color="#e74c3c", opacity=0.7,
                                       xbins=dict(start=1, end=12, size=1)))
        fig_len.update_layout(
            barmode="overlay", title="Journey Length Distribution",
            xaxis_title="Number of Touchpoints",
            yaxis_title="Count", height=340,
            template="plotly_white", legend=dict(x=0.65, y=0.95),
        )
        st.plotly_chart(fig_len, width='stretch')

    st.markdown("---")
    st.subheader("📊 Online vs Offline Channel Mix")
    online_pct = []
    for j in journeys:
        if j["n_touches"] > 0:
            online = sum(1 for c in j["path"] if CHANNEL_TYPE[c] == "Online")
            online_pct.append(online / j["n_touches"] * 100)

    conv_mix    = [p for p, j in zip(online_pct, journeys) if j["converted"]]
    nonconv_mix = [p for p, j in zip(online_pct, journeys) if not j["converted"]]

    fig_mix = go.Figure()
    fig_mix.add_trace(go.Box(y=conv_mix,    name="Converted",     marker_color="#27ae60"))
    fig_mix.add_trace(go.Box(y=nonconv_mix, name="Not Converted", marker_color="#e74c3c"))
    fig_mix.update_layout(
        title="% Online Touchpoints by Conversion Outcome",
        yaxis_title="% Online Touchpoints",
        height=350, template="plotly_white",
    )
    st.plotly_chart(fig_mix, width='stretch')


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — BUDGET OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_budget:
    st.subheader("💰 Shapley-Driven Budget Optimizer")
    st.markdown("""
    Attribution weights from **Shapley values** are used as the basis for a
    **constrained budget reallocation** problem. The optimizer maximises expected
    conversions (modelled with diminishing-returns response curves: $α·\\text{spend}^{0.5}$)
    subject to total budget and per-channel min/max constraints.
    """)

    if "Shapley" not in attr_df.columns:
        st.warning("Run Shapley model first (enabled by default).")
    else:
        sh_weights = {
            ch: float(attr_df.loc[CHANNEL_LABELS[ch], "Shapley"])
            for ch in CHANNELS if CHANNEL_LABELS[ch] in attr_df.index
        }

        # Current spend from CPT-proportional baseline
        cpt_vals = np.array([CHANNEL_CPT[ch] + 1 for ch in CHANNELS], dtype=float)
        curr_weights = cpt_vals / cpt_vals.sum()
        current_spend = {ch: total_budget * w for ch, w in zip(CHANNELS, curr_weights)}

        with st.spinner("Optimising budget allocation…"):
            opt_df = optimize_budget(
                sh_weights, total_budget,
                min_per_channel=min_alloc,
                max_per_channel=max_alloc,
                current_spend=current_spend,
            )

        col_l, col_r = st.columns(2)
        with col_l:
            st.plotly_chart(budget_waterfall(opt_df), width='stretch')
        with col_r:
            st.plotly_chart(budget_delta_chart(opt_df), width='stretch')

        st.markdown("---")
        st.subheader("Allocation Details")
        disp_opt = opt_df[[
            "channel_label", "attribution_weight",
            "current_spend", "optimised_spend", "delta", "delta_pct"
        ]].copy()
        disp_opt.columns = ["Channel","Attribution Weight","Current ($)","Optimised ($)","Delta ($)","Delta (%)"]
        disp_opt["Attribution Weight"] = (disp_opt["Attribution Weight"] * 100).round(1).astype(str) + "%"
        disp_opt["Current ($)"]    = disp_opt["Current ($)"].apply(lambda x: f"${x:,.0f}")
        disp_opt["Optimised ($)"]  = disp_opt["Optimised ($)"].apply(lambda x: f"${x:,.0f}")
        disp_opt["Delta ($)"]      = disp_opt["Delta ($)"].apply(lambda x: f"${x:+,.0f}")
        disp_opt["Delta (%)"]      = disp_opt["Delta (%)"].apply(lambda x: f"{x:+.1f}%")
        st.dataframe(disp_opt, width='stretch', hide_index=True)

        total_lift = opt_df["response_lift"].sum()
        total_curr_resp = opt_df["current_response"].sum()
        pct_lift = total_lift / max(total_curr_resp, 1e-9) * 100
        st.success(
            f"📈 **Reallocation lifts expected conversions by {pct_lift:.1f}%** "
            f"(response function units: {total_curr_resp:.3f} → {total_curr_resp + total_lift:.3f})"
        )

        st.info(
            "⚠️ **Note:** Response curves use a stylised $α·\\text{spend}^{0.5}$ model. "
            "In production, fit channel-specific response curves from historical experiments or MMM."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — MARKOV ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_markov:
    st.subheader("📈 Markov Chain Attribution")
    st.markdown("""
    The **Markov chain model** (Anderl et al., 2016) builds a transition probability
    matrix across channel states. Attribution credit is the **removal effect**:
    how much the baseline conversion probability drops when a channel is removed from the graph
    (its inbound transitions redirected to a null/loss state).
    """)

    # Prefer the already-computed Markov values from run_all_models (cached).
    # Fall back to load_markov only if Markov was disabled in the sidebar.
    if "Markov Chain" in attr_df.columns:
        markov_vals = {
            ch: float(attr_df.loc[CHANNEL_LABELS[ch], "Markov Chain"])
            for ch in CHANNELS if CHANNEL_LABELS[ch] in attr_df.index
        }
    else:
        with st.spinner("Computing Markov transition matrix…"):
            markov_vals = load_markov(journeys_hash, journeys)

    col_heat, col_bar = st.columns([3, 2])
    with col_heat:
        st.plotly_chart(markov_transition_heatmap(journeys), width='stretch')

    with col_bar:
        st.subheader("Removal Effect Attribution")
        mkv_df = pd.DataFrame({
            "Channel": [CHANNEL_LABELS[ch] for ch in CHANNELS],
            "Removal Effect": [markov_vals.get(ch, 0.0) * 100 for ch in CHANNELS],
        }).sort_values("Removal Effect", ascending=True)

        fig_mkv = go.Figure(go.Bar(
            x=mkv_df["Removal Effect"],
            y=mkv_df["Channel"],
            orientation="h",
            marker_color="#3498db",
            text=mkv_df["Removal Effect"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
        ))
        fig_mkv.update_layout(
            title="Markov Removal Effect (%)",
            xaxis_title="Attribution Credit (%)",
            height=420,
            template="plotly_white",
            margin=dict(l=130, t=50),
        )
        st.plotly_chart(fig_mkv, width='stretch')

    st.markdown("---")
    st.subheader("Markov vs Shapley Comparison")
    if "Shapley" in attr_df.columns and "Markov Chain" in attr_df.columns:
        comp = attr_df[["Shapley", "Markov Chain"]].copy() * 100
        fig_comp = go.Figure()
        x = comp.index.tolist()
        fig_comp.add_trace(go.Scatter(
            x=x, y=comp["Shapley"], mode="lines+markers",
            name="Shapley", line=dict(color="#9b59b6", width=2),
            marker=dict(size=8),
        ))
        fig_comp.add_trace(go.Scatter(
            x=x, y=comp["Markov Chain"], mode="lines+markers",
            name="Markov Chain", line=dict(color="#3498db", width=2, dash="dash"),
            marker=dict(size=8, symbol="diamond"),
        ))
        fig_comp.update_layout(
            title="Shapley vs Markov: Attribution Credit per Channel (%)",
            yaxis_title="Attribution Credit (%)",
            height=380, template="plotly_white",
        )
        fig_comp.update_xaxes(tickangle=-25)
        st.plotly_chart(fig_comp, width='stretch')

        corr = comp["Shapley"].corr(comp["Markov Chain"])
        st.metric("Shapley ↔ Markov correlation", f"{corr:.3f}",
                  help="How closely Shapley and Markov agree on channel rankings.")

    st.markdown("---")
    with st.expander("📚 Markov model details"):
        st.markdown("""
        **States:** Each channel is a state, plus absorbing states `CONVERSION` and `NULL`.

        **Transition counting:** For every journey `[ch₁, ch₂, …, chₙ]`, we count
        transitions `ch₁→ch₂`, `ch₂→ch₃`, …, `chₙ₋₁→chₙ`, then `chₙ→CONV/NULL`.

        **Conversion probability:** Solved analytically via the fundamental matrix
        $\\mathbf{f} = (I - Q)^{-1}\\mathbf{r}$ where $Q$ is the transient sub-matrix
        and $\\mathbf{r}$ is the absorption probability vector.

        **Removal effect:** For channel $i$, set all transitions *to* $i$ to go to NULL instead.
        The drop in conversion probability is channel $i$'s attribution credit.
        """)



# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8 — MMM + MTA HYBRID
# ═══════════════════════════════════════════════════════════════════════════════
with tab_mmm:

    # ── Pull correct variables from MTA ─────────────────────────────────────
    total_conv = sum(1 for j in journeys if j["converted"])
    total_rev  = sum(j["value"] for j in journeys)

    shapley_w = (
        {ch: float(attr_df.loc[CHANNEL_LABELS[ch], "Shapley"])
         for ch in CHANNELS if CHANNEL_LABELS[ch] in attr_df.index}
        if "Shapley" in attr_df.columns else {}
    )
    mmm_contributions = {k: v for k, v in CHANNEL_TRUE_CONTRIBUTION.items() if k != "base"}

    # ── Alpha slider in sidebar ──────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔀 Hybrid MMM+MTA Settings")
    alpha = st.sidebar.slider(
        "MTA Weight α (online channels)",
        min_value=0.0, max_value=1.0, value=DEFAULT_ALPHA, step=0.05,
        help="α=1 → pure MTA, α=0 → pure MMM. Offline channels always use 100% MMM.",
    )

    # ── Page header ──────────────────────────────────────────────────────────
    st.markdown("## 🔀 Hybrid MMM + MTA Attribution")

    st.info("""
    **Why this tab?**
    Your existing MTA models (Shapley, Markov) can only credit channels that appear in
    **digital journey logs**. Offline channels — TV, Radio, Direct Mail, Agent Visits —
    are **invisible to MTA** even though they genuinely drive conversions upstream.

    **Marketing Mix Modelling (MMM)** fixes this by measuring offline impact at the
    **weekly aggregate level** using regression with adstock + saturation transformations.
    This tab blends both models into one unified framework with a **common metric
    (CPIC & Marginal ROI)** so every channel can be compared on the same axis.
    """)

    # ── Load MMM data (cached) ───────────────────────────────────────────────
    with st.spinner("Generating 104-week synthetic MMM dataset..."):
        mmm_df, mmm_meta = load_mmm_data(n_weeks=104, seed=42)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1 — What is MMM and how does it work?
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📖 Section 1 — How MMM Works")

    col_explain1, col_explain2, col_explain3 = st.columns(3)
    with col_explain1:
        st.markdown("""
        **Step 1 — Collect Weekly Data**

        Instead of tracking individual users, MMM works at the **aggregate level**:
        - How much did we spend on TV this week?
        - How many radio impressions ran?
        - How many direct mail pieces were sent?
        - What were total conversions that week?

        External factors like seasonality, holidays, and competitor activity
        are also included as control variables.
        """)
    with col_explain2:
        st.markdown("""
        **Step 2 — Apply Adstock & Saturation**

        Two transformations make the model realistic:

        **Adstock (carryover):** A TV ad aired Monday still influences
        purchases on Thursday. Each channel has a *decay rate* λ:
        `adstock_t = spend_t + λ × adstock_(t-1)`

        **Saturation:** Doubling spend doesn't double conversions.
        A Hill function captures this: after a point, more spend
        delivers diminishing returns.
        """)
    with col_explain3:
        st.markdown("""
        **Step 3 — Fit Regression & Extract Contributions**

        A Bayesian regression model estimates how much each channel's
        (adstock-transformed) spend contributed to total conversions.

        Output: **contribution share per channel** — e.g.,
        "TV drove 18% of this quarter's conversions."

        This is what MTA *cannot* tell you for offline channels.
        The Hybrid model then combines MMM offline shares with
        MTA online shares into one unified weight.
        """)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2 — MMM Data Overview
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Section 2 — Synthetic MMM Dataset (104 Weeks)")

    st.markdown("""
    The dataset below simulates **2 years of weekly marketing data** for all channels.
    Each row = one ISO week. This is the kind of data you'd feed into a real MMM model.
    """)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Weeks of Data",      f"{len(mmm_df)}")
    m2.metric("Total Conversions",  f"{mmm_df['conversions'].sum():,}")
    m3.metric("Total Revenue",      f"${mmm_df['revenue'].sum()/1e6:.1f}M")
    m4.metric("Avg Weekly Conv.",   f"{mmm_df['conversions'].mean():.0f}")
    m5.metric("Peak Week Conv.",    f"{mmm_df['conversions'].max()}")

    with st.expander("📋 View raw MMM weekly data (first 10 rows)", expanded=False):
        display_cols = [
            "iso_week", "tv_grp", "tv_spend", "radio_impressions_000",
            "radio_spend", "direct_mail_pieces", "direct_mail_spend",
            "agent_visits", "agent_visit_spend",
            "paid_search_spend", "seasonality_index",
            "is_holiday_week", "conversions", "revenue"
        ]
        st.dataframe(
            mmm_df[display_cols].head(10).style.format({
                "tv_spend": "${:,.0f}", "radio_spend": "${:,.0f}",
                "direct_mail_spend": "${:,.0f}", "agent_visit_spend": "${:,.0f}",
                "paid_search_spend": "${:,.0f}", "revenue": "${:,.0f}",
                "seasonality_index": "{:.3f}",
            }),
            width='stretch',
        )
        st.caption("tv_grp = Gross Rating Points | radio_impressions_000 = impressions in thousands | direct_mail_pieces = pieces mailed | agent_visits = visits count")

    # Spend by channel bar
    summary_mmm = mmm_summary_stats(mmm_df)
    fig_spend = px.bar(
        summary_mmm, x="Channel", y="Total Spend ($)", color="Type",
        color_discrete_map={"Offline": "#e6550d", "Online": "#1f77b4"},
        title="Total Spend Over 2 Years — Offline vs Online Channels",
        text_auto=".2s",
    )
    fig_spend.update_layout(height=380, plot_bgcolor="white",
                            xaxis_title="", yaxis_title="Total Spend ($)")
    st.plotly_chart(fig_spend, width='stretch')

    # Active weeks info
    st.markdown("""
    > **Notice:** Offline channels (TV, Radio, Direct Mail) show **fewer than 104 active weeks**.
    > This is intentional — offline media is typically run in **flights** (bursts), not continuously.
    > This is what makes MMM necessary: the adstock model tracks carryover effect
    > *between* flights, which MTA completely misses.
    """)

    # Conversions over time + seasonality
    st.markdown("#### Weekly Conversions with Seasonality & Holiday Effects")
    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(
        x=mmm_df["week_start_date"], y=mmm_df["conversions"],
        name="Weekly Conversions", line=dict(color="#2196F3", width=1.8),
        fill="tozeroy", fillcolor="rgba(33,150,243,0.08)",
    ))
    fig_conv.add_trace(go.Scatter(
        x=mmm_df["week_start_date"],
        y=(mmm_df["seasonality_index"] * mmm_df["conversions"].mean()).round(0),
        name="Seasonality Trend (scaled)", line=dict(color="#e6550d", dash="dot", width=2),
    ))
    holiday_wks = mmm_df[mmm_df["is_holiday_week"] == 1]
    fig_conv.add_trace(go.Scatter(
        x=holiday_wks["week_start_date"], y=holiday_wks["conversions"],
        mode="markers", name="Holiday Week",
        marker=dict(color="gold", size=9, symbol="star",
                    line=dict(color="#333", width=1)),
    ))
    fig_conv.update_layout(
        height=380, plot_bgcolor="white",
        title="Conversions spike on holiday weeks (Diwali, Christmas, New Year) and follow annual seasonality",
        xaxis_title="Week", yaxis_title="Conversions",
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig_conv, width='stretch')

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3 — Adstock: The Carryover Effect
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📉 Section 3 — Adstock: Why Offline Is Hard to Credit")

    st.markdown("""
    The core reason MTA fails for offline channels is **adstock** — the carryover effect.
    When a TV campaign ends, its influence doesn't stop immediately. Consumers remember the ad
    for days or weeks. This means:

    - A customer who converts "organically" on Day 14 may have been influenced by a TV ad on Day 1
    - MTA sees: `Organic Search → Conversion` and credits Organic Search
    - Reality: TV planted the intent, Organic Search just captured it

    The chart below shows the gap between **actual spend** (bars) and **adstock-adjusted influence** (line).
    The line stays elevated even after spend drops to zero — that's the carryover.
    """)

    ch_to_plot = st.selectbox(
        "Select channel to explore adstock carryover:",
        options=["tv", "radio", "direct_mail", "agent_visit"],
        format_func=lambda x: CHANNEL_LABELS.get(x, x),
        key="adstock_selector"
    )

    spend_col   = f"{ch_to_plot}_spend"
    adstock_col = f"{ch_to_plot}_adstock"
    decay_val   = ADSTOCK_DECAY[ch_to_plot]

    fig_adstock = go.Figure()
    fig_adstock.add_trace(go.Bar(
        x=mmm_df["week_start_date"], y=mmm_df[spend_col],
        name="Actual Weekly Spend ($)", marker_color="#aec7e8", opacity=0.65,
    ))
    fig_adstock.add_trace(go.Scatter(
        x=mmm_df["week_start_date"], y=mmm_df[adstock_col],
        name=f"Adstock (λ={decay_val})", line=dict(color="#e6550d", width=2.5),
    ))
    fig_adstock.update_layout(
        height=380, plot_bgcolor="white",
        title=f"{CHANNEL_LABELS.get(ch_to_plot)} — Actual Spend vs Adstock-Adjusted Influence",
        xaxis_title="Week", yaxis_title="$",
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig_adstock, width='stretch')

    st.caption(
        f"λ={decay_val} means {decay_val*100:.0f}% of last week's influence carries into this week. "
        f"When spend = $0 but adstock > 0, the channel is still driving conversions — "
        f"MTA records those as 'direct' or 'organic' since no offline touchpoint is logged."
    )

    # Decay rate comparison table
    st.markdown("#### Adstock Decay Rates by Channel")
    decay_df = pd.DataFrame([
        {"Channel": CHANNEL_LABELS.get(ch, ch),
         "Type": "Offline" if ch in ("tv","radio","direct_mail","agent_visit") else "Online",
         "Decay Rate (λ)": v,
         "Half-Life (weeks)": round(-1 / np.log(v), 1) if v > 0 else 0,
         "Interpretation": (
             "Very long memory — ~3-4 weeks" if v >= 0.6 else
             "Long memory — ~2-3 weeks" if v >= 0.4 else
             "Moderate — ~1-2 weeks" if v >= 0.2 else
             "Short — days only"
         )}
        for ch, v in ADSTOCK_DECAY.items()
    ]).sort_values("Decay Rate (λ)", ascending=False)

    st.dataframe(
        decay_df.style
        .background_gradient(subset=["Decay Rate (λ)"], cmap="OrRd")
        .format({"Decay Rate (λ)": "{:.2f}", "Half-Life (weeks)": "{:.1f}"}),
        width='stretch', hide_index=True,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4 — The Offline Credit Gap
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Section 4 — The Offline Credit Gap: MTA vs MMM")

    st.markdown("""
    This is the core problem this tab solves. The chart below shows the same 10 channels
    under two different lenses:
    - **MTA (Shapley)** — credits channels based on digital journey data only
    - **MMM** — credits channels based on aggregate contribution to conversions (sees offline)

    Offline channels will appear **undercredited in MTA** and **correctly credited in MMM**.
    """)

    if not shapley_w:
        st.warning("Shapley model not computed. Enable it in the sidebar and re-run.")
    else:
        recovery = offline_credit_recovery(shapley_w, mmm_contributions)
        comparison_df = compare_mta_vs_mmm_vs_hybrid(shapley_w, mmm_contributions, alpha)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("MTA Offline Credit",    f"{recovery['mta_offline_credit_pct']}%",
                  help="TV + Radio + Direct Mail + Agent Visit share in MTA")
        r2.metric("MMM Offline Credit",    f"{recovery['mmm_offline_credit_pct']}%",
                  help="True offline contribution measured by MMM")
        r3.metric("Undercredit Gap",       f"+{recovery['undercredit_gap_pct']}pp",
                  delta=f"MTA misses {recovery['undercredit_gap_pct']}pp of offline value",
                  delta_color="inverse")
        r4.metric("MTA Online Overcredit", f"+{recovery['undercredit_gap_pct']}pp",
                  delta="Online channels absorb offline's missing credit",
                  delta_color="inverse")

        # Side-by-side grouped bar
        fig_gap = go.Figure()
        model_config = [
            ("MTA (Shapley)", "#9b59b6", "mta_pct"),
            ("MMM (True Contribution)", "#e6550d", "mmm_pct"),
            (f"Hybrid (α={alpha:.2f})", "#27ae60", "hybrid_pct"),
        ]
        for name, color, col in model_config:
            fig_gap.add_trace(go.Bar(
                x=comparison_df["channel_label"],
                y=comparison_df[col],
                name=name, marker_color=color,
                text=comparison_df[col].apply(lambda x: f"{x:.1f}%"),
                textposition="outside",
                textfont=dict(size=9),
            ))
        fig_gap.update_layout(
            barmode="group", height=440,
            title="Attribution Credit per Channel: MTA vs MMM vs Hybrid",
            xaxis_title="Channel", yaxis_title="Credit Share (%)",
            plot_bgcolor="white",
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig_gap, width='stretch')

        # Credit gap table
        with st.expander("📋 Full Credit Comparison Table — showing the undercredit/overcredit gap"):
            gap_display = comparison_df[[
                "channel_label", "channel_type", "mta_pct", "mmm_pct", "hybrid_pct", "credit_gap"
            ]].rename(columns={
                "channel_label": "Channel", "channel_type": "Type",
                "mta_pct": "MTA %", "mmm_pct": "MMM %",
                "hybrid_pct": "Hybrid %", "credit_gap": "Gap (MMM−MTA) pp",
            })

            def color_gap(val):
                if val > 1:   return "color: #e74c3c; font-weight:bold"
                if val < -1:  return "color: #27ae60; font-weight:bold"
                return ""

            st.dataframe(
                gap_display.style
                .applymap(color_gap, subset=["Gap (MMM−MTA) pp"])
                .format({"MTA %": "{:.1f}", "MMM %": "{:.1f}",
                         "Hybrid %": "{:.1f}", "Gap (MMM−MTA) pp": "{:+.1f}"}),
                width='stretch', hide_index=True,
            )
        st.caption("🔴 Red gap = MTA undercredits this channel vs MMM  |  🟢 Green = MTA overcredits")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5 — Hybrid Blend
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 🧬 Section 5 — Hybrid Blend (α = {alpha:.2f})")

    st.markdown(f"""
    The blending formula is simple but powerful:

    | Channel Type | Formula |
    |---|---|
    | **Online** (Paid Search, Display, etc.) | `Hybrid = {alpha:.0%} × MTA + {1-alpha:.0%} × MMM` |
    | **Offline** (TV, Radio, Direct Mail, Agent Visit) | `Hybrid = 100% × MMM` |

    **Why pure MMM for offline?** Because MTA gives offline channels ~0% (they're invisible),
    so blending in MTA would only dilute the correct MMM signal. Offline always uses MMM fully.

    **Adjust the α slider** in the left sidebar to see how the blend shifts.
    Higher α = trust MTA more for online. Lower α = trust MMM more for everything.
    """)

    if shapley_w:
        hybrid_weights = blend_mta_mmm(shapley_w, mmm_contributions, alpha)

        col_donut, col_bar = st.columns([1, 1])

        with col_donut:
            fig_donut = go.Figure(go.Pie(
                labels=[CHANNEL_LABELS.get(ch, ch) for ch in hybrid_weights],
                values=[v * 100 for v in hybrid_weights.values()],
                hole=0.45,
                marker_colors=[CHANNEL_COLORS.get(ch, "#999") for ch in hybrid_weights],
                textinfo="label+percent",
                textfont=dict(size=11),
            ))
            fig_donut.update_layout(
                height=420,
                title=f"Hybrid Attribution Distribution (α={alpha:.2f})",
                annotations=[dict(text="Hybrid", x=0.5, y=0.5,
                                  font_size=15, showarrow=False)],
            )
            st.plotly_chart(fig_donut, width='stretch')

        with col_bar:
            # Online vs Offline share comparison across 3 models
            offline_chs = {"tv", "radio", "direct_mail", "agent_visit"}
            mta_total = sum(shapley_w.values()) or 1
            mmm_total = sum(mmm_contributions.values()) or 1
            h_total   = sum(hybrid_weights.values()) or 1

            mta_off  = sum(shapley_w.get(c,0) for c in offline_chs) / mta_total * 100
            mmm_off  = sum(mmm_contributions.get(c,0) for c in offline_chs) / mmm_total * 100
            hyb_off  = sum(hybrid_weights.get(c,0) for c in offline_chs) / h_total * 100

            fig_split = go.Figure()
            for label, on_val, off_val, color in [
                ("MTA",    100-mta_off, mta_off, "#9b59b6"),
                ("MMM",    100-mmm_off, mmm_off, "#e6550d"),
                (f"Hybrid\n(α={alpha:.2f})", 100-hyb_off, hyb_off, "#27ae60"),
            ]:
                fig_split.add_trace(go.Bar(
                    name="Online", x=[label], y=[on_val],
                    marker_color="#1f77b4", showlegend=(label=="MTA"),
                    text=f"{on_val:.1f}%", textposition="inside",
                ))
                fig_split.add_trace(go.Bar(
                    name="Offline", x=[label], y=[off_val],
                    marker_color="#e6550d", showlegend=(label=="MTA"),
                    text=f"{off_val:.1f}%", textposition="inside",
                ))
            fig_split.update_layout(
                barmode="stack", height=420,
                title="Online vs Offline Credit Split by Model",
                yaxis_title="Credit Share (%)",
                plot_bgcolor="white",
                legend=dict(orientation="h", y=1.05),
            )
            st.plotly_chart(fig_split, width='stretch')

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 6 — Unified Common Metric
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💡 Section 6 — Unified Common Metric: CPIC & Marginal ROI")

    st.markdown("""
    The final step: convert everything to a single comparable metric so TV and
    Paid Search can be evaluated on the same axis.

    - **CPIC (Cost Per Incremental Conversion):** `Total Spend ÷ Hybrid-Attributed Conversions` — lower is better
    - **Marginal ROI:** `Hybrid-Attributed Revenue ÷ Spend` — higher is better, break-even = 1.0x

    These metrics let you answer: *"If I shift $10k from TV to Paid Search, do I gain or lose?"*
    """)

    if shapley_w:
        unified_df = compute_unified_metrics(
            hybrid_weights,
            total_conversions=float(total_conv),
            total_revenue=float(total_rev),
        )

        col_cpic, col_roi = st.columns(2)

        with col_cpic:
            fig_cpic = px.bar(
                unified_df.sort_values("cpic"),
                x="cpic", y="channel_label",
                orientation="h",
                color="channel_type",
                color_discrete_map={"Offline": "#e6550d", "Online": "#1f77b4"},
                title="CPIC — Cost Per Incremental Conversion (Lower = Better)",
                labels={"cpic": "CPIC ($)", "channel_label": ""},
                text_auto="$.0f",
            )
            fig_cpic.update_layout(height=400, plot_bgcolor="white",
                                   showlegend=False)
            st.plotly_chart(fig_cpic, width='stretch')

        with col_roi:
            fig_mroi = px.bar(
                unified_df.sort_values("marginal_roi", ascending=True),
                x="marginal_roi", y="channel_label",
                orientation="h",
                color="channel_type",
                color_discrete_map={"Offline": "#e6550d", "Online": "#1f77b4"},
                title="Marginal ROI by Channel (Higher = Better, 1.0x = Break-even)",
                labels={"marginal_roi": "Marginal ROI (x)", "channel_label": ""},
                text_auto=".2f",
            )
            fig_mroi.add_vline(x=1.0, line_dash="dash", line_color="red",
                               annotation_text="Break-even", annotation_position="top right")
            fig_mroi.update_layout(height=400, plot_bgcolor="white",
                                   showlegend=False)
            st.plotly_chart(fig_mroi, width='stretch')

        # Full unified table
        st.markdown("#### 📋 Full Unified Metrics Table")
        st.markdown("All channels on the same scale — online and offline, fairly compared.")
        st.dataframe(
            unified_df[[
                "channel_label", "channel_type", "hybrid_weight",
                "attributed_conversions", "spend", "cpic", "marginal_roi"
            ]].rename(columns={
                "channel_label": "Channel", "channel_type": "Type",
                "hybrid_weight": "Credit Share (%)",
                "attributed_conversions": "Attributed Conversions",
                "spend": "Spend ($)", "cpic": "CPIC ($)", "marginal_roi": "Marginal ROI",
            }).style
            .format({
                "Credit Share (%)": "{:.1f}%",
                "Attributed Conversions": "{:.0f}",
                "Spend ($)": "${:,.0f}",
                "CPIC ($)": "${:,.0f}",
                "Marginal ROI": "{:.2f}x",
            })
            .background_gradient(subset=["Marginal ROI"], cmap="RdYlGn")
            .background_gradient(subset=["CPIC ($)"], cmap="RdYlGn_r"),
            width='stretch', hide_index=True,
        )

        st.success("""
        💡 **How to read this table:**
        Channels with **high Marginal ROI** and **low CPIC** are your most efficient spends.
        Channels that appear strong in MTA but drop in the Hybrid view were being
        *overcredited* by MTA (stealing credit from offline channels upstream).
        """)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 7 — Ground Truth Validation
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ✅ Section 7 — Ground-Truth Validation (Synthetic Data Advantage)")
    st.markdown("""
    One major benefit of using **synthetic data** for MMM development is that
    we know the *true* contribution of each channel — it's baked into the data generator.
    In production, you'd compare your MMM-recovered contributions against these
    to validate your model before trusting it with real budget decisions.
    """)

    truth_df = pd.DataFrame([
        {
            "Channel": CHANNEL_LABELS.get(ch, ch),
            "Type": "Offline" if ch in ("tv","radio","direct_mail","agent_visit") else "Online",
            "True Contribution %": round(v * 100, 1),
            "Adstock Decay (λ)": ADSTOCK_DECAY.get(ch, "N/A"),
            "Carryover Half-Life": f"{round(-1/np.log(ADSTOCK_DECAY[ch]),1)}w" if ch in ADSTOCK_DECAY else "N/A",
        }
        for ch, v in CHANNEL_TRUE_CONTRIBUTION.items() if ch != "base"
    ]).sort_values("True Contribution %", ascending=False)

    fig_truth = px.bar(
        truth_df, x="Channel", y="True Contribution %", color="Type",
        color_discrete_map={"Offline": "#e6550d", "Online": "#1f77b4"},
        title="Ground-Truth Channel Contributions (Baked Into Synthetic DGP)",
        text_auto=".1f",
    )
    fig_truth.update_layout(height=380, plot_bgcolor="white",
                            xaxis_title="", yaxis_title="True Contribution (%)")
    st.plotly_chart(fig_truth, width='stretch')

    st.dataframe(
        truth_df.style
        .background_gradient(subset=["True Contribution %"], cmap="Blues")
        .format({"True Contribution %": "{:.1f}%", "Adstock Decay (λ)": "{:.2f}"}),
        width='stretch', hide_index=True,
    )
    st.caption(
        "After fitting a real Bayesian MMM (e.g., PyMC-Marketing), the recovered "
        "contribution coefficients should be close to the 'True Contribution %' column. "
        "Large deviations indicate model misspecification or insufficient data."
    )


st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#95a5a6; font-size:0.8rem; padding:1rem 0">
    Multi-Touch Attribution Demo · Cooperative Game Theory (Shapley, Banzhaf, Markov) ·
    GBT Characteristic Function · Plackett-Luce Ordered Shapley · Bootstrap CIs ·
    Built with Streamlit &amp; Plotly ·
    <em>Zhao et al. (2018) · Grabisch &amp; Roubens (1999) · Anderl et al. (2016)</em>
</div>
""", unsafe_allow_html=True)
