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
import base64

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


# ── Logo loader ───────────────────────────────────────────────────────────────
def get_logo_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_b64 = get_logo_base64("logo 1.jpg")


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
        ordered_n_samples=ord_samples,
    )

@st.cache_data(show_spinner=False)
def load_interactions(journeys_hash, _journeys):
    return shapley_interaction_index(_journeys)

@st.cache_data(show_spinner=False)
def load_markov(journeys_hash, _journeys):
    return markov_chain(_journeys)

@st.cache_data(show_spinner=False)
def load_bootstrap_ci(journeys_hash, _journeys, n_boot):
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
st.markdown(f"""
<div style="
    background: linear-gradient(90deg, #2c3e50, #4b6cb7);
    padding: 0 28px;
    border-radius: 12px;
    margin-bottom: 20px;
    color: white;
    display: flex;
    align-items: stretch;
    min-height: 90px;
    overflow: hidden;
">
    <!-- Logo strip on left -->
    <div style="
        background: white;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0 18px;
        margin: 0 24px 0 -28px;
        min-width: 110px;
        flex-shrink: 0;
    ">
        <img src="data:image/jpeg;base64,{logo_b64}"
             style="width: 80px; height: 80px; object-fit: contain; display: block;" />
    </div>
    <!-- Text block -->
    <div style="
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 18px 0;
    ">
        <h2 style="margin: 0 0 6px 0; font-size: 1.55rem; font-weight: 700; line-height: 1.2;">
            Marketing Attribution Intelligence Platform
        </h2>
        <p style="margin: 0 0 4px 0; font-size: 15px; opacity: 0.95;">
            Unified measurement across digital and offline marketing channels
        </p>
        <p style="margin: 0; font-size: 12.5px; opacity: 0.82;">
            Shapley Attribution &nbsp;•&nbsp; Markov Modeling &nbsp;•&nbsp; Marketing Mix Modeling &nbsp;•&nbsp; Budget Optimization
        </p>
    </div>
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
tab_overview, tab_compare, tab_shapley, tab_synergy, tab_journey, tab_budget, tab_markov, tab_mmm, tab_scenario = st.tabs([
    "📊 Overview",
    "🎯 Model Comparison",
    "🔬 Shapley Deep Dive",
    "🔗 Channel Synergies",
    "🛤️ Journey Explorer",
    "💰 Budget Optimizer",
    "📈 Markov Analysis",
    "🔀 MMM + MTA Hybrid",
    "🧪 Scenario Planner",
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
    The **Shapley Interaction Index** measures whether two channels are *synergistic*
    (work better together than separately) or *substitutable* (overlap in the journeys they drive).

    **Red** = synergy (channels reinforce each other) · **Blue** = substitution (channels overlap)
    """)

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


    # ── Load MMM data (cached) ───────────────────────────────────────────────
    with st.spinner("Generating 104-week synthetic MMM dataset..."):
        mmm_df, mmm_meta = load_mmm_data(n_weeks=104, seed=42)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1 — MMM Data Overview
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Section 1 — Synthetic MMM Dataset (104 Weeks)")

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
    # SECTION 2 — Adstock: The Carryover Effect
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📉 Section 2 — Adstock: Why Offline Is Hard to Credit")

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
    # SECTION 3 — The Offline Credit Gap
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Section 3 — The Offline Credit Gap: MTA vs MMM")

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
    # SECTION 4 — Hybrid Blend
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 🧬 Section 4 — Hybrid Blend (α = {alpha:.2f})")

    st.markdown(f"""
    The blending formula combines MTA and MMM weights by channel type:

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
    # SECTION 5 — Unified Common Metric
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💡 Section 5 — Unified Common Metric: CPIC & Marginal ROI")

    st.markdown("""
    Convert everything to a single comparable metric so TV and Paid Search can be
    evaluated on the same axis.

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
            fig_cpic.update_layout(height=400, plot_bgcolor="white", showlegend=False)
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
            fig_mroi.update_layout(height=400, plot_bgcolor="white", showlegend=False)
            st.plotly_chart(fig_mroi, width='stretch')

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
    # SECTION 6 — Ground Truth Validation
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ✅ Section 6 — Ground-Truth Validation (Synthetic Data Advantage)")
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


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 9 — SCENARIO PLANNER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_scenario:

    # ── Helper: response curve ────────────────────────────────────────────────
    def _resp(spend: float, alpha: float, beta: float = 0.5) -> float:
        """Diminishing-returns response: alpha * spend^beta."""
        return alpha * (spend ** beta) if spend > 0 else 0.0

    # ── Pull Shapley weights (already computed above) ─────────────────────────
    if "Shapley" not in attr_df.columns:
        st.warning("⚠️ Shapley model not computed. Enable it in the sidebar.")
        st.stop()

    sh_w = {
        ch: float(attr_df.loc[CHANNEL_LABELS[ch], "Shapley"])
        for ch in CHANNELS if CHANNEL_LABELS[ch] in attr_df.index
    }
    attr_arr   = np.array([sh_w.get(ch, 1e-6) for ch in CHANNELS])
    attr_arr   = np.maximum(attr_arr, 1e-6)
    alpha_arr  = attr_arr / attr_arr.sum()   # response-curve alphas

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("## 🧪 Scenario Planner — What-If Budget Simulator")
    st.markdown(
        "Model the impact of **increasing or decreasing your total budget** "
        "and see exactly which channels should absorb the change, "
        "with projected conversion lift and statistical confidence bands."
    )

    with st.expander("💡 How to use this tab", expanded=False):
        st.markdown(
            "1. Set your **current** and **new** budget in Section 1.\n"
            "2. Choose **Auto-Optimise** to let Shapley weights decide the distribution, "
            "or **Manual** to drag sliders yourself.\n"
            "3. Read the lift estimate and confidence band in Section 3.\n"
            "4. Save up to 4 named scenarios and compare them side-by-side in Section 4.\n\n"
            "> ⚠️ *Response curves use a stylised α·spend^0.5 model. "
            "For production, replace with MMM-fitted saturation curves from the Hybrid tab.*"
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Scenario Setup
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📐 Section 1 — Scenario Setup")

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        sc_current_budget = st.number_input(
            "Current Total Budget ($)",
            min_value=10_000, max_value=5_000_000,
            value=int(total_budget),
            step=10_000,
            help="Match this to the Budget Optimizer tab for consistency.",
        )
    with col_s2:
        sc_new_budget = st.number_input(
            "New Total Budget ($)",
            min_value=10_000, max_value=5_000_000,
            value=int(total_budget * 1.10),
            step=10_000,
            help="Set higher for growth scenarios, lower for budget cut scenarios.",
        )
    with col_s3:
        alloc_mode = st.radio(
            "Distribution Mode",
            ["🤖 Auto-Optimise", "🎛️ Manual Sliders"],
            help=(
                "Auto: Shapley-weighted optimal split of the delta. "
                "Manual: You decide how to distribute the budget change."
            ),
        )

    sc_delta = sc_new_budget - sc_current_budget
    delta_color = "#27ae60" if sc_delta >= 0 else "#e74c3c"
    delta_label = f"{'📈 Budget Increase' if sc_delta >= 0 else '📉 Budget Cut'}"

    # ── Compute baseline current spend per channel (CPT-weighted, used for Section 3) ──
    cpt_vals    = np.array([CHANNEL_CPT[ch] + 1 for ch in CHANNELS], dtype=float)
    curr_w      = cpt_vals / cpt_vals.sum()
    sc_curr_spend = {ch: sc_current_budget * w for ch, w in zip(CHANNELS, curr_w)}

    # ── Auto-optimise new allocation (always computed, used in auto mode) ─────
    @st.cache_data(show_spinner=False)
    def _scenario_optimise(sh_weights_hash, new_budget, curr_spend_vals,
                           _sh_w, _curr_spend, min_a, max_a):
        return optimize_budget(
            _sh_w, new_budget,
            min_per_channel=min_a,
            max_per_channel=max_a,
            current_spend=_curr_spend,
        )

    auto_opt_df = _scenario_optimise(
        hash(tuple(sorted(sh_w.items()))),
        sc_new_budget,
        tuple(sc_curr_spend[ch] for ch in CHANNELS),
        sh_w, sc_curr_spend, min_alloc, max_alloc,
    )
    auto_spend = dict(zip(auto_opt_df["channel"], auto_opt_df["optimised_spend"]))

    # ── Shapley-optimised baseline at CURRENT budget (for uplift banner) ──────
    # This is what the optimizer would allocate if given the current budget —
    # uplift = going from this optimised baseline to the new-budget optimised allocation
    @st.cache_data(show_spinner=False)
    def _scenario_optimise_curr(sh_weights_hash, curr_budget, curr_spend_vals,
                                _sh_w, _curr_spend, min_a, max_a):
        return optimize_budget(
            _sh_w, curr_budget,
            min_per_channel=min_a,
            max_per_channel=max_a,
            current_spend=_curr_spend,
        )

    curr_opt_df   = _scenario_optimise_curr(
        hash(tuple(sorted(sh_w.items()))),
        sc_current_budget,
        tuple(sc_curr_spend[ch] for ch in CHANNELS),
        sh_w, sc_curr_spend, min_alloc, max_alloc,
    )
    curr_opt_spend = dict(zip(curr_opt_df["channel"], curr_opt_df["optimised_spend"]))

    # ── Quick conversion uplift preview for banner (auto mode only) ───────────
    attr_arr_b  = np.array([sh_w.get(ch, 1e-6) for ch in CHANNELS])
    attr_arr_b  = np.maximum(attr_arr_b, 1e-6)
    alpha_arr_b = attr_arr_b / attr_arr_b.sum()

    # Baseline = Shapley-optimised at current budget
    _total_curr_resp_b = sum(_resp(curr_opt_spend.get(ch, sc_curr_spend[ch]), alpha_arr_b[i])
                             for i, ch in enumerate(CHANNELS))
    _total_conv_b      = float(sum(1 for j in journeys if j["converted"]))
    _sf_b              = _total_conv_b / max(_total_curr_resp_b, 1e-9)
    _curr_conv_b       = _total_curr_resp_b * _sf_b

    # New = Shapley-optimised at new budget
    _auto_conv_b   = sum(_resp(auto_spend.get(ch, sc_curr_spend[ch]), alpha_arr_b[i])
                         for i, ch in enumerate(CHANNELS)) * _sf_b
    _uplift_conv_b = _auto_conv_b - _curr_conv_b
    _uplift_pct_b  = _uplift_conv_b / max(_curr_conv_b, 1) * 100
    _uplift_color_b = "#27ae60" if _uplift_conv_b >= 0 else "#e74c3c"
    _uplift_sign_b  = "+" if _uplift_conv_b >= 0 else ""

    # Build uplift pill as a plain string (avoids double-interpolation in outer f-string)
    _is_auto = "🤖 Auto-Optimise" in alloc_mode
    if _is_auto:
        _curr_conv_str   = f"{_curr_conv_b:,.0f}"
        _auto_conv_str   = f"{_auto_conv_b:,.0f}"
        _uplift_val_str  = f"{_uplift_sign_b}{_uplift_conv_b:.0f} ({_uplift_sign_b}{_uplift_pct_b:.1f}%)"
        _conv_uplift_html = (
            "<div style='border-left:2px solid #dee2e6;padding-left:30px;display:flex;gap:24px;align-items:center'>"
            # current conv
            "<div>"
            "<div style='font-size:0.75rem;color:#6c757d'>Current Conv.</div>"
            "<div style='font-size:1.4rem;font-weight:700'>" + _curr_conv_str + "</div>"
            "</div>"
            # arrow
            "<div style='font-size:1.8rem;color:" + _uplift_color_b + "'>&#8594;</div>"
            # new conv
            "<div>"
            "<div style='font-size:0.75rem;color:#6c757d'>New Conv.</div>"
            "<div style='font-size:1.4rem;font-weight:700'>" + _auto_conv_str + "</div>"
            "</div>"
            # uplift
            "<div style='border-left:2px solid #dee2e6;padding-left:24px'>"
            "<div style='font-size:0.75rem;color:#6c757d'>&#127919; Conv. Uplift</div>"
            "<div style='font-size:1.4rem;font-weight:700;color:" + _uplift_color_b + "'>" + _uplift_val_str + "</div>"
            "</div>"
            "</div>"
        )
    else:
        _conv_uplift_html = ""

    # ── Manual conv uplift for banner (needs confirmed spend from session state) ─
    if "🎛️ Manual" in alloc_mode and "sc_confirmed_spend" in st.session_state:
        _mc_spend = st.session_state["sc_confirmed_spend"]
        _mc_alpha = np.maximum(np.array([sh_w.get(ch, 1e-6) for ch in CHANNELS]), 1e-6)
        _mc_alpha = _mc_alpha / _mc_alpha.sum()
        _mc_curr_resp = sum(_resp(curr_opt_spend.get(ch, sc_curr_spend[ch]), _mc_alpha[i])
                            for i, ch in enumerate(CHANNELS))
        _mc_sf        = float(sum(1 for j in journeys if j["converted"])) / max(_mc_curr_resp, 1e-9)
        _mc_curr_conv = _mc_curr_resp * _mc_sf
        _mc_new_conv  = sum(_resp(_mc_spend.get(ch, 0), _mc_alpha[i])
                            for i, ch in enumerate(CHANNELS)) * _mc_sf
        _mc_lift      = _mc_new_conv - _mc_curr_conv
        _mc_lift_pct  = _mc_lift / max(_mc_curr_conv, 1) * 100
        _mc_color     = "#27ae60" if _mc_lift >= 0 else "#e74c3c"
        _mc_sign      = "+" if _mc_lift >= 0 else ""
        _mc_conv_html = (
            "<div style='border-left:2px solid #dee2e6;padding-left:30px;display:inline-flex;gap:40px;align-items:center'>"
            "<div><div style='font-size:0.75rem;color:#6c757d'>Current Conv.</div>"
            "<div style='font-size:1.4rem;font-weight:700'>" + f"{_mc_curr_conv:,.0f}" + "</div></div>"
            "<div style='font-size:1.8rem;color:" + _mc_color + "'>&#8594;</div>"
            "<div><div style='font-size:0.75rem;color:#6c757d'>New Conv.</div>"
            "<div style='font-size:1.4rem;font-weight:700'>" + f"{_mc_new_conv:,.0f}" + "</div></div>"
            "<div style='border-left:2px solid #dee2e6;padding-left:30px'>"
            "<div style='font-size:0.75rem;color:#6c757d'>&#127919; Conv. Uplift</div>"
            "<div style='font-size:1.4rem;font-weight:700;color:" + _mc_color + "'>"
            + f"{_mc_sign}{_mc_lift:.0f} ({_mc_sign}{_mc_lift_pct:.1f}%)" +
            "</div></div></div>"
        )
    else:
        _mc_conv_html = ""

    # ── Delta banner — fit-content so it shrinks to text width ───────────────
    _curr_budget_str  = f"${sc_current_budget:,.0f}"
    _new_budget_str   = f"${sc_new_budget:,.0f}"
    _delta_str        = f"${sc_delta:+,.0f} ({sc_delta/sc_current_budget*100:+.1f}%)"

    st.markdown(
        "<div style='"
        "background:linear-gradient(135deg,#f8f9fa,#e9ecef);"
        "border-left:5px solid " + delta_color + ";"
        "border-radius:8px;padding:14px 20px;margin:12px 0;"
        "display:inline-flex;gap:40px;align-items:center;min-width:0;"
        "'>"
        "<div>"
        "<div style='font-size:0.75rem;color:#6c757d'>Current Budget</div>"
        "<div style='font-size:1.4rem;font-weight:700'>" + _curr_budget_str + "</div>"
        "</div>"
        "<div style='font-size:1.8rem;color:" + delta_color + "'>&#8594;</div>"
        "<div>"
        "<div style='font-size:0.75rem;color:#6c757d'>New Budget</div>"
        "<div style='font-size:1.4rem;font-weight:700'>" + _new_budget_str + "</div>"
        "</div>"
        "<div style='border-left:2px solid #dee2e6;padding-left:30px'>"
        "<div style='font-size:0.75rem;color:#6c757d'>" + delta_label + "</div>"
        "<div style='font-size:1.4rem;font-weight:700;color:" + delta_color + "'>" + _delta_str + "</div>"
        "</div>"
        + _conv_uplift_html
        + _mc_conv_html +
        "</div>",
        unsafe_allow_html=True,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Allocation (Auto or Manual)
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")

    if "🤖 Auto-Optimise" in alloc_mode:
        st.markdown("### 🤖 Section 2 — Optimal Allocation (Shapley-Weighted)")
        st.markdown(
            "The optimizer distributes the **full new budget** using Shapley-weighted "
            "diminishing-returns response curves. Channels that are least saturated "
            "relative to their attribution weight absorb the most incremental spend."
        )

        st.markdown(
            "> 💡 **Tip:** Channels with high Shapley weight AND low current spend "
            "have the steepest marginal return — that's where your extra budget has "
            "the highest ROI right now."
        )

        # Side-by-side current vs optimal — both Shapley-optimised, different budgets
        fig_auto = go.Figure()
        ch_labels_plot = [CHANNEL_LABELS[ch] for ch in CHANNELS]
        curr_vals = [curr_opt_spend.get(ch, sc_curr_spend[ch]) for ch in CHANNELS]
        opt_vals  = [auto_spend.get(ch, sc_curr_spend[ch]) for ch in CHANNELS]

        fig_auto.add_trace(go.Bar(
            name="Current Spend", x=ch_labels_plot, y=curr_vals,
            marker_color="#95a5a6",
            text=[f"${v:,.0f}" for v in curr_vals], textposition="outside",
            textfont=dict(size=9),
        ))
        fig_auto.add_trace(go.Bar(
            name="Optimal New Spend", x=ch_labels_plot, y=opt_vals,
            marker_color="#9b59b6",
            text=[f"${v:,.0f}" for v in opt_vals], textposition="outside",
            textfont=dict(size=9),
        ))
        fig_auto.update_layout(
            barmode="group", height=420, template="plotly_white",
            title=f"Current vs Optimal Spend — New Budget ${sc_new_budget:,.0f}",
            yaxis_title="Spend ($)", xaxis_title="",
            legend=dict(orientation="h", y=1.08),
            margin=dict(t=70, b=60),
        )
        fig_auto.update_xaxes(tickangle=-25)
        st.plotly_chart(fig_auto, width='stretch')

        # Delta bar
        deltas = [opt_vals[i] - curr_vals[i] for i in range(len(CHANNELS))]
        fig_delta = go.Figure(go.Bar(
            x=ch_labels_plot, y=deltas,
            marker_color=["#27ae60" if d >= 0 else "#e74c3c" for d in deltas],
            text=[f"${d:+,.0f}" for d in deltas], textposition="outside",
            textfont=dict(size=9),
        ))
        fig_delta.add_hline(y=0, line_color="#333", line_width=1.2)
        fig_delta.update_layout(
            height=340, template="plotly_white",
            title="Budget Delta per Channel (New Optimal − Current)",
            yaxis_title="Change ($)", xaxis_title="",
            margin=dict(t=60, b=60),
        )
        fig_delta.update_xaxes(tickangle=-25)
        st.plotly_chart(fig_delta, width='stretch')

        # Which channel absorbs most of the increment — explain why
        if sc_delta != 0:
            delta_arr = np.array(deltas)
            top_ch_idx = int(np.argmax(np.abs(delta_arr)))
            top_ch     = CHANNELS[top_ch_idx]
            top_label  = CHANNEL_LABELS[top_ch]
            top_delta  = deltas[top_ch_idx]
            top_sh_pct = sh_w.get(top_ch, 0) * 100
            st.success(
                f"🔍 **Key insight:** **{top_label}** absorbs the largest share of the budget change "
                f"(${top_delta:+,.0f}) because it holds the highest Shapley weight "
                f"({top_sh_pct:.1f}%) relative to its current saturation level. "
                f"Its marginal return on additional spend is steeper than competing channels."
            )

        sc_new_spend = auto_spend

    else:
        # ── Manual sliders ────────────────────────────────────────────────────
        st.markdown("### 🎛️ Section 2 — Manual Budget Allocation")
        st.markdown(
            "Drag sliders to manually distribute the **new total budget** across channels. "
            "Online and offline channels are separated into distinct panels. "
            "The budget tracker updates live — green means fully allocated."
        )
        st.markdown(
            "> 💡 **Tip:** Start by matching the auto-optimised allocation "
            "as a benchmark, then override specific channels to test your hypothesis. "
            "Watch the projected lift in Section 3 update in real time."
        )

        default_auto = {ch: auto_spend.get(ch, sc_curr_spend[ch]) for ch in CHANNELS}
        online_chs  = [ch for ch in CHANNELS if CHANNEL_TYPE[ch] == "Online"]
        offline_chs = [ch for ch in CHANNELS if CHANNEL_TYPE[ch] == "Offline"]

        # ── live budget tracker (top) ─────────────────────────────────────────
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg,#1a1a2e,#16213e);
            border-radius:12px; padding:16px 24px; margin:0 0 18px 0;
            display:inline-flex; align-items:center; gap:32px; flex-wrap:wrap;
        ">
            <div>
                <div style="font-size:0.7rem;color:#adb5bd;letter-spacing:.08em;
                            text-transform:uppercase">Total Budget</div>
                <div style="font-size:1.5rem;font-weight:800;color:#fff">
                    ${sc_new_budget:,.0f}</div>
            </div>
            <div style="height:36px;width:1px;background:#444"></div>
            <div>
                <div style="font-size:0.7rem;color:#adb5bd;letter-spacing:.08em;
                            text-transform:uppercase">Online Channels</div>
                <div style="font-size:1.1rem;font-weight:700;color:#4fc3f7">
                    ${sum(default_auto[ch] for ch in online_chs):,.0f}</div>
            </div>
            <div>
                <div style="font-size:0.7rem;color:#adb5bd;letter-spacing:.08em;
                            text-transform:uppercase">Offline Channels</div>
                <div style="font-size:1.1rem;font-weight:700;color:#ffb74d">
                    ${sum(default_auto[ch] for ch in offline_chs):,.0f}</div>
            </div>
            <div style="height:36px;width:1px;background:#444"></div>
            <div style="font-size:0.75rem;color:#adb5bd;font-style:italic">
                Adjust sliders below — tracker updates on rerun
            </div>
        </div>
        """, unsafe_allow_html=True)

        manual_allocs = {}

        # ── channel slider helper ─────────────────────────────────────────────
        def _channel_slider(ch, accent_color):
            ch_label   = CHANNEL_LABELS[ch]
            sh_pct     = sh_w.get(ch, 0) * 100
            curr_sp    = sc_curr_spend[ch]
            default_v  = int(round(default_auto[ch] / 1000) * 1000)
            max_v      = int(sc_new_budget * 0.60)  # cap at 60% per channel

            # Render Shapley pill first (static)
            st.markdown(f"""
            <div style="
                display:flex; justify-content:space-between;
                align-items:center; margin-bottom:2px;
            ">
                <span style="font-weight:600;font-size:0.88rem;color:#212529">
                    {ch_label}
                </span>
                <span style="
                    background:{accent_color}22;
                    color:{accent_color};
                    border:1px solid {accent_color}55;
                    border-radius:20px;padding:1px 8px;
                    font-size:0.7rem;font-weight:700
                ">Shapley {sh_pct:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

            val = st.slider(
                label=f"__{ch_label}__",
                min_value=0,
                max_value=max_v,
                value=min(default_v, max_v),
                step=1_000,
                key=f"slider_{ch}",
                label_visibility="collapsed",
                help=f"Shapley weight: {sh_pct:.1f}%  |  "
                     f"Current spend: ${curr_sp:,.0f}  |  "
                     f"Auto-optimal: ${default_auto[ch]:,.0f}",
            )

            # Live delta pill — computed AFTER slider renders using actual val
            live_delta     = val - curr_sp
            live_sign      = "▲" if live_delta >= 0 else "▼"
            live_col       = "#4caf50" if live_delta >= 0 else "#ef5350"

            # spend display pill + live delta below slider
            st.markdown(f"""
            <div style="
                display:flex;justify-content:space-between;align-items:center;
                margin-top:-8px;margin-bottom:10px;
                font-size:0.75rem;color:#6c757d
            ">
                <span>$0</span>
                <span style="display:flex;gap:8px;align-items:center">
                    <span style="
                        background:{accent_color};color:#fff;
                        border-radius:6px;padding:1px 10px;
                        font-weight:700;font-size:0.8rem
                    ">${val:,.0f}</span>
                    <span style="color:{live_col};font-weight:600;font-size:0.75rem">
                        {live_sign} ${abs(live_delta):,.0f}</span>
                </span>
                <span>${max_v:,.0f}</span>
            </div>
            """, unsafe_allow_html=True)

            return val

        # ── ONLINE container ──────────────────────────────────────────────────
        st.markdown(f"""
        <div style="
            border:2px solid #1976d2;border-radius:14px;
            padding:18px 22px 8px 22px;margin-bottom:20px;
            background:linear-gradient(135deg,#e3f2fd 0%,#fafeff 100%);
        ">
        <div style="
            display:flex;align-items:center;gap:10px;margin-bottom:16px
        ">
            <span style="
                background:#1976d2;color:white;border-radius:8px;
                padding:3px 14px;font-size:0.8rem;font-weight:700;
                letter-spacing:.06em;text-transform:uppercase
            ">🌐 Online Channels</span>
            <span style="font-size:0.78rem;color:#555">
                Digital — trackable in MTA journeys
            </span>
        </div>
        """, unsafe_allow_html=True)

        oc1, oc2 = st.columns(2)
        for idx, ch in enumerate(online_chs):
            with (oc1 if idx % 2 == 0 else oc2):
                manual_allocs[ch] = _channel_slider(ch, "#1976d2")

        st.markdown("</div>", unsafe_allow_html=True)

        # ── OFFLINE container ─────────────────────────────────────────────────
        st.markdown(f"""
        <div style="
            border:2px solid #e65100;border-radius:14px;
            padding:18px 22px 8px 22px;margin-bottom:20px;
            background:linear-gradient(135deg,#fff3e0 0%,#fffaf6 100%);
        ">
        <div style="
            display:flex;align-items:center;gap:10px;margin-bottom:16px
        ">
            <span style="
                background:#e65100;color:white;border-radius:8px;
                padding:3px 14px;font-size:0.8rem;font-weight:700;
                letter-spacing:.06em;text-transform:uppercase
            ">📺 Offline Channels</span>
            <span style="font-size:0.78rem;color:#555">
                TV · Radio · Direct Mail · Agent Visit — adstock-driven, use MMM for precision
            </span>
        </div>
        """, unsafe_allow_html=True)

        fc1, fc2 = st.columns(2)
        for idx, ch in enumerate(offline_chs):
            with (fc1 if idx % 2 == 0 else fc2):
                manual_allocs[ch] = _channel_slider(ch, "#e65100")

        st.markdown("</div>", unsafe_allow_html=True)

        # ── live budget tracker (bottom) ──────────────────────────────────────
        total_manual    = sum(manual_allocs.values())
        remaining       = sc_new_budget - total_manual
        alloc_pct       = min(total_manual / max(sc_new_budget, 1) * 100, 100)
        rem_color       = "#27ae60" if abs(remaining) < 1_000 else (
                          "#f39c12" if abs(remaining) < 5_000 else "#e74c3c")
        bar_color       = "#27ae60" if abs(remaining) < 1_000 else (
                          "#f39c12" if abs(remaining) < 5_000 else "#3498db")

        online_total  = sum(manual_allocs[ch] for ch in online_chs)
        offline_total = sum(manual_allocs[ch] for ch in offline_chs)

        st.markdown(f"""
        <div style="
            background:#f8f9fa;border-radius:12px;
            padding:16px 22px;margin:4px 0 6px 0;
            border:1px solid #dee2e6;
        ">
            <!-- progress bar -->
            <div style="
                height:8px;border-radius:99px;
                background:#dee2e6;margin-bottom:12px;overflow:hidden
            ">
                <div style="
                    height:100%;width:{alloc_pct:.1f}%;
                    background:{bar_color};border-radius:99px;
                    transition:width .3s ease
                "></div>
            </div>
            <!-- row of stats -->
            <div style="display:flex;gap:28px;flex-wrap:wrap;align-items:center">
                <div>
                    <div style="font-size:0.68rem;color:#6c757d;text-transform:uppercase;
                                letter-spacing:.07em">Allocated</div>
                    <div style="font-size:1.1rem;font-weight:700;color:#212529">
                        ${total_manual:,.0f}</div>
                </div>
                <div>
                    <div style="font-size:0.68rem;color:#6c757d;text-transform:uppercase;
                                letter-spacing:.07em">Remaining</div>
                    <div style="font-size:1.1rem;font-weight:700;color:{rem_color}">
                        ${remaining:+,.0f}</div>
                </div>
                <div>
                    <div style="font-size:0.68rem;color:#6c757d;text-transform:uppercase;
                                letter-spacing:.07em">🌐 Online</div>
                    <div style="font-size:1rem;font-weight:600;color:#1976d2">
                        ${online_total:,.0f}
                        ({online_total/max(total_manual,1)*100:.0f}%)</div>
                </div>
                <div>
                    <div style="font-size:0.68rem;color:#6c757d;text-transform:uppercase;
                                letter-spacing:.07em">📺 Offline</div>
                    <div style="font-size:1rem;font-weight:600;color:#e65100">
                        ${offline_total:,.0f}
                        ({offline_total/max(total_manual,1)*100:.0f}%)</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        sc_new_spend = manual_allocs

        # ── Compute button + budget gate ──────────────────────────────────────
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        budget_ok = abs(remaining) < 1_000   # within $1k tolerance

        if not budget_ok:
            over_under = "over-allocated" if remaining < 0 else "under-allocated"
            st.markdown(f"""
            <div style="
                background:#fff8e1;border:1.5px solid #f39c12;
                border-radius:10px;padding:14px 20px;margin-bottom:12px;
                display:flex;align-items:center;gap:14px;
            ">
                <span style="font-size:1.6rem">⚠️</span>
                <div>
                    <div style="font-weight:700;color:#b7770d;font-size:0.95rem">
                        Budget not fully allocated</div>
                    <div style="color:#7d5a00;font-size:0.82rem;margin-top:2px">
                        You are <b>{over_under}</b> by
                        <b>${abs(remaining):,.0f}</b>
                        (${total_manual:,.0f} of ${sc_new_budget:,.0f} used).
                        Adjust the sliders above until the tracker turns green,
                        then click <b>Compute Impact</b>.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        col_btn, _ = st.columns([1, 3])
        with col_btn:
            compute_clicked = st.button(
                "⚡ Compute Impact",
                disabled=not budget_ok,
                type="primary",
                use_container_width=True,
                help="Fully allocate the budget first (tracker must be green).",
            )

        # Hash of current slider state — used to detect changes after compute
        current_slider_hash = hash(tuple(
            st.session_state.get(f"slider_{ch}", 0) for ch in CHANNELS
        ) + (sc_new_budget,))

        # Persist the confirmed allocation in session state
        if compute_clicked and budget_ok:
            st.session_state["sc_confirmed_spend"]       = dict(sc_new_spend)
            st.session_state["sc_confirmed_budget"]      = sc_new_budget
            st.session_state["sc_confirmed_slider_hash"] = current_slider_hash
            st.rerun()

        # Invalidate results if sliders have changed since last compute
        if st.session_state.get("sc_confirmed_slider_hash") != current_slider_hash:
            st.session_state.pop("sc_confirmed_spend",       None)
            st.session_state.pop("sc_confirmed_budget",      None)
            st.session_state.pop("sc_confirmed_slider_hash", None)

        # Gate Section 3+ on confirmed spend existing
        _confirmed = st.session_state.get("sc_confirmed_spend")
        if _confirmed is None:
            st.markdown("""
            <div style="
                background:#f0f4ff;border:1.5px dashed #9b59b6;
                border-radius:12px;padding:30px;margin-top:18px;
                text-align:center;color:#6c757d;
            ">
                <div style="font-size:2rem;margin-bottom:8px">📊</div>
                <div style="font-weight:600;font-size:1rem;color:#4a235a">
                    Results will appear here</div>
                <div style="font-size:0.85rem;margin-top:4px">
                    Fully allocate the budget and click
                    <b>⚡ Compute Impact</b> to see projections,
                    confidence bands, and the scenario comparison.
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.stop()

        # Use the confirmed (locked) allocation for all downstream computation
        sc_new_spend = _confirmed

    # If auto mode, always allow Section 3 (no gate needed)
    if "🤖 Auto-Optimise" in alloc_mode:
        if "sc_confirmed_spend" in st.session_state:
            del st.session_state["sc_confirmed_spend"]

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Projected Impact + Statistical Significance
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📊 Section 3 — Projected Conversion Impact & Statistical Significance")

    st.markdown(
        "Response curves (α·spend^0.5) translate spend into expected conversions. "
        "Confidence bands are derived from **Shapley bootstrap CIs** — "
        "the uncertainty in attribution propagates into the conversion projections. "
        "Enable Bootstrap CIs in the sidebar (Shapley Deep Dive tab) for tighter bands."
    )

    # Compute current and new responses per channel
    # Baseline = Shapley-optimised at current budget (consistent with banner)
    _baseline_spend = curr_opt_spend  # Shapley-optimal at sc_current_budget
    curr_responses = {}
    new_responses  = {}
    for i, ch in enumerate(CHANNELS):
        a = alpha_arr[i]
        curr_responses[ch] = _resp(_baseline_spend.get(ch, sc_curr_spend[ch]), a)
        new_responses[ch]  = _resp(sc_new_spend.get(ch, sc_curr_spend[ch]), a)

    total_curr_resp = sum(curr_responses.values())
    total_new_resp  = sum(new_responses.values())
    total_lift_resp = total_new_resp - total_curr_resp

    # Scale response units to real conversions from MTA data
    total_conv_float = float(sum(1 for j in journeys if j["converted"]))
    scale_factor     = total_conv_float / max(total_curr_resp, 1e-9)

    proj_curr_conv = total_curr_resp * scale_factor
    proj_new_conv  = total_new_resp  * scale_factor
    proj_lift_conv = proj_new_conv - proj_curr_conv
    proj_lift_pct  = proj_lift_conv / max(proj_curr_conv, 1) * 100

    # ── Confidence band: propagate Shapley CI uncertainty if available ────────
    ci_available = run_ci and "ci_df" in dir()
    try:
        if run_ci:
            ci_df_sp = load_bootstrap_ci(journeys_hash, journeys, n_bootstrap)
            ci_available = True
        else:
            ci_df_sp = None
    except Exception:
        ci_df_sp = None
        ci_available = False

    # Build per-channel CI bands on response
    ch_ci_width = {}
    if ci_available and ci_df_sp is not None:
        for _, row in ci_df_sp.iterrows():
            ch_ci_width[row["channel"]] = float(row["ci_width"])
    else:
        # Fallback: assume ±15% CI width on each Shapley weight
        for ch in CHANNELS:
            ch_ci_width[ch] = sh_w.get(ch, 0.05) * 0.30

    # Propagate CI through response curve (linear approximation)
    lift_variance = 0.0
    for i, ch in enumerate(CHANNELS):
        a        = alpha_arr[i]
        new_sp   = sc_new_spend.get(ch, sc_curr_spend[ch])
        curr_sp  = sc_curr_spend[ch]
        # d(response)/d(alpha) at current allocation
        if new_sp > 0:
            dR_da    = new_sp ** 0.5
        else:
            dR_da    = 0.0
        ci_w     = ch_ci_width.get(ch, 0.02)
        lift_variance += (dR_da * ci_w) ** 2

    lift_std_resp    = lift_variance ** 0.5
    lift_std_conv    = lift_std_resp * scale_factor
    ci_lower_conv    = proj_lift_conv - 1.96 * lift_std_conv
    ci_upper_conv    = proj_lift_conv + 1.96 * lift_std_conv

    # Statistical significance: is the lift > 0 at 95% confidence?
    is_significant   = ci_lower_conv > 0 if proj_lift_conv > 0 else ci_upper_conv < 0
    z_score          = proj_lift_conv / max(lift_std_conv, 1e-9)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    kpi_data = [
        (k1, "Current Conversions",  f"{proj_curr_conv:,.0f}", "#3498db"),
        (k2, "Conversion Lift",       f"{proj_lift_conv:+,.0f} ({proj_lift_pct:+.1f}%)",
             "#27ae60" if proj_lift_conv >= 0 else "#e74c3c"),
        (k3, "95% CI Range",          f"{ci_lower_conv:+.0f} to {ci_upper_conv:+.0f}", "#e67e22"),
        (k4, "Statistically Significant",
             "✅ Yes" if is_significant else "⚠️ No",
             "#27ae60" if is_significant else "#e74c3c"),
    ]
    for col, label, val, color in kpi_data:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-color:{color}">
                <h3>{label}</h3>
                <h2 style="color:{color};font-size:1.2rem">{val}</h2>
            </div>""", unsafe_allow_html=True)

    if not is_significant:
        st.warning(
            f"⚠️ **Lift is not statistically significant at 95% confidence.** "
            f"The projected lift of {proj_lift_conv:+.0f} conversions falls within "
            f"the attribution noise range (z={z_score:.2f}). "
            f"Consider a larger budget change or enabling Bootstrap CIs in the sidebar "
            f"for more precise uncertainty estimates."
        )
    else:
        st.success(
            f"✅ **Lift is statistically significant** (z={z_score:.2f}). "
            f"The projected +{proj_lift_conv:.0f} conversions lie outside the 95% "
            f"confidence band — this is a reliably detectable improvement."
        )



    # ── Per-channel lift waterfall ─────────────────────────────────────────────
    ch_lifts     = [(CHANNEL_LABELS[ch],
                     (new_responses[ch] - curr_responses[ch]) * scale_factor,
                     CHANNEL_COLORS.get(ch, "#aaa"))
                    for ch in CHANNELS]
    ch_lifts_s   = sorted(ch_lifts, key=lambda x: -x[1])
    lift_labels  = [x[0] for x in ch_lifts_s]
    lift_vals    = [x[1] for x in ch_lifts_s]
    lift_colors  = [x[2] for x in ch_lifts_s]

    # CI band per channel (proportional share of total CI)
    ch_lift_ci   = []
    for ch in sorted(CHANNELS, key=lambda c: -(new_responses[c] - curr_responses[c])):
        share = abs(new_responses[ch] - curr_responses[ch]) / max(abs(total_lift_resp), 1e-9)
        ch_lift_ci.append(1.96 * lift_std_conv * share)

    fig_lift = go.Figure()
    fig_lift.add_trace(go.Bar(
        x=lift_labels, y=lift_vals,
        marker_color=lift_colors,
        error_y=dict(type="data", array=ch_lift_ci, visible=True,
                     color="#4a235a", thickness=2, width=6),
        text=[f"{v:+.0f}" for v in lift_vals],
        textposition="outside",
        name="Projected Lift",
    ))
    fig_lift.add_hline(y=0, line_color="#333", line_width=1.2)
    fig_lift.update_layout(
        height=400, template="plotly_white",
        title="Projected Conversion Lift per Channel (with 95% CI whiskers)",
        yaxis_title="Incremental Conversions",
        xaxis_title="",
        margin=dict(t=60, b=60),
    )
    fig_lift.update_xaxes(tickangle=-25)
    st.plotly_chart(fig_lift, width='stretch')

    # ── Response curve visualiser ─────────────────────────────────────────────
    st.markdown("#### 📈 Spend-Response Curves — Current vs New Spend Point")
    st.markdown(
        "> 💡 **Reading this chart:** Each curve shows diminishing returns for one channel. "
        "The dot marks where you are now; the triangle marks where the new budget puts you. "
        "Channels with steep curves at the new point still have room to grow."
    )

    rc_ch = st.selectbox(
        "Channel to inspect",
        [CHANNEL_LABELS[ch] for ch in CHANNELS],
        key="rc_ch_select",
    )
    rc_ch_key = next(ch for ch in CHANNELS if CHANNEL_LABELS[ch] == rc_ch)
    rc_idx    = CHANNELS.index(rc_ch_key)
    rc_alpha  = alpha_arr[rc_idx]
    rc_color  = CHANNEL_COLORS.get(rc_ch_key, "#9b59b6")

    _rc_curr_sp = _baseline_spend.get(rc_ch_key, sc_curr_spend[rc_ch_key])
    x_max = max(_rc_curr_sp, sc_new_spend.get(rc_ch_key, 0)) * 1.5 + 5_000
    x_range = np.linspace(0, x_max, 300)
    y_range = [_resp(x, rc_alpha) * scale_factor for x in x_range]

    fig_rc = go.Figure()
    fig_rc.add_trace(go.Scatter(
        x=x_range, y=y_range,
        mode="lines", name="Response Curve",
        line=dict(color=rc_color, width=2.5),
    ))
    # Current point (Shapley-optimised at current budget)
    curr_pt_y = _resp(_rc_curr_sp, rc_alpha) * scale_factor
    fig_rc.add_trace(go.Scatter(
        x=[_rc_curr_sp], y=[curr_pt_y],
        mode="markers", name="Current Spend",
        marker=dict(color="#3498db", size=14, symbol="circle",
                    line=dict(color="white", width=2)),
    ))
    # New point
    new_sp_rc = sc_new_spend.get(rc_ch_key, _rc_curr_sp)
    new_pt_y  = _resp(new_sp_rc, rc_alpha) * scale_factor
    fig_rc.add_trace(go.Scatter(
        x=[new_sp_rc], y=[new_pt_y],
        mode="markers", name="New Spend",
        marker=dict(color="#27ae60", size=14, symbol="triangle-up",
                    line=dict(color="white", width=2)),
    ))
    # Lift annotation arrow
    fig_rc.add_annotation(
        x=new_sp_rc, y=new_pt_y,
        ax=sc_curr_spend[rc_ch_key], ay=curr_pt_y,
        xref="x", yref="y", axref="x", ayref="y",
        arrowhead=3, arrowwidth=2, arrowcolor=delta_color,
        text=f"{new_pt_y - curr_pt_y:+.1f} conv",
        font=dict(size=11, color=delta_color),
        showarrow=True,
    )
    fig_rc.update_layout(
        height=370, template="plotly_white",
        title=f"{rc_ch} — Spend-Response Curve (α·spend^0.5, scaled to actual conversions)",
        xaxis_title="Spend ($)", yaxis_title="Expected Conversions",
        legend=dict(orientation="h", y=1.05),
        margin=dict(t=70, b=50),
    )
    st.plotly_chart(fig_rc, width='stretch')

    # ── Marginal ROI table ────────────────────────────────────────────────────
    st.markdown("#### 📋 Channel-Level Projections")
    st.markdown(
        "> 💡 **Marginal ROI** = projected revenue from new conversions ÷ additional spend. "
        "Values > 1.0x mean you're getting more revenue than you're putting in at the margin."
    )

    avg_order_val_sc = float(sum(j["value"] for j in journeys)) / max(
        float(sum(1 for j in journeys if j["converted"])), 1
    )

    proj_rows = []
    for i, ch in enumerate(CHANNELS):
        curr_sp  = _baseline_spend.get(ch, sc_curr_spend[ch])
        new_sp   = sc_new_spend.get(ch, curr_sp)
        sp_delta = new_sp - curr_sp
        c_resp   = curr_responses[ch] * scale_factor
        n_resp   = new_responses[ch]  * scale_factor
        lift     = n_resp - c_resp
        mroi     = (lift * avg_order_val_sc) / max(abs(sp_delta), 1) if sp_delta != 0 else 0.0
        proj_rows.append({
            "Channel":         CHANNEL_LABELS[ch],
            "Type":            CHANNEL_TYPE[ch],
            "ch_key":          ch,
            "Shapley Wt (%)":  round(sh_w.get(ch, 0) * 100, 1),
            "Current Spend":   round(curr_sp, 0),
            "New Spend":       round(new_sp, 0),
            "sp_delta_raw":    round(sp_delta, 0),
            "Curr Conv (est)": round(c_resp, 1),
            "New Conv (est)":  round(n_resp, 1),
            "lift_raw":        round(lift, 1),
            "mroi_raw":        round(mroi, 2),
        })

    proj_df = pd.DataFrame(proj_rows).sort_values("lift_raw", ascending=False)

    # ── colour helpers ────────────────────────────────────────────────────────
    def _spend_delta_color(v):
        return "#27ae60" if v >= 0 else "#e74c3c"

    def _lift_color(v):
        return "#27ae60" if v >= 0 else "#e74c3c"

    def _mroi_bg(v):
        if v > 1.5:  return "#d5f5e3"
        if v > 1.0:  return "#eafaf1"
        if v < 0:    return "#fde8e8"
        return "#ffffff"

    def _mroi_fg(v):
        if v > 1.5:  return "#1e8449"
        if v > 1.0:  return "#27ae60"
        if v < 0:    return "#e74c3c"
        return "#555555"

    def _type_badge(t):
        if t == "Online":
            return ("<span style='background:#e3f2fd;color:#1976d2;"
                    "border:1px solid #90caf9;border-radius:20px;"
                    "padding:1px 9px;font-size:0.72rem;font-weight:600'>"
                    "Online</span>")
        return ("<span style='background:#fff3e0;color:#e65100;"
                "border:1px solid #ffcc80;border-radius:20px;"
                "padding:1px 9px;font-size:0.72rem;font-weight:600'>"
                "Offline</span>")

    def _shapley_bar(pct):
        bar_w = min(int(pct * 4), 60)
        return (f"<div style='display:flex;align-items:center;gap:6px'>"
                f"<div style='width:{bar_w}px;height:7px;border-radius:4px;"
                f"background:#9b59b6;opacity:0.75'></div>"
                f"<span style='font-size:0.82rem;color:#555'>{pct:.1f}%</span>"
                f"</div>")

    # ── build HTML table ──────────────────────────────────────────────────────
    header_style = ("background:#f8f9fa;font-size:0.75rem;font-weight:700;"
                    "color:#6c757d;text-transform:uppercase;letter-spacing:.06em;"
                    "padding:10px 14px;border-bottom:2px solid #dee2e6;white-space:nowrap")
    cell_style   = "padding:9px 14px;font-size:0.85rem;border-bottom:1px solid #f0f0f0;vertical-align:middle"

    headers = ["Channel", "Type", "Shapley Wt", "Current Spend",
               "New Spend", "Δ Spend", "Curr Conv", "New Conv", "Conv Lift", "Marginal ROI"]

    rows_html = ""
    for _, row in proj_df.iterrows():
        sd_col  = _spend_delta_color(row["sp_delta_raw"])
        lf_col  = _lift_color(row["lift_raw"])
        mr_bg   = _mroi_bg(row["mroi_raw"])
        mr_fg   = _mroi_fg(row["mroi_raw"])
        sp_sign = "+" if row["sp_delta_raw"] >= 0 else ""
        lf_sign = "+" if row["lift_raw"] >= 0 else ""
        ch_color = CHANNEL_COLORS.get(row["ch_key"], "#aaa")

        rows_html += f"""
        <tr style='background:#fff' onmouseover="this.style.background='#fafafa'"
            onmouseout="this.style.background='#fff'">
            <td style='{cell_style}'>
                <div style='display:flex;align-items:center;gap:7px'>
                    <div style='width:10px;height:10px;border-radius:3px;
                                background:{ch_color};flex-shrink:0'></div>
                    <span style='font-weight:600;color:#212529'>{row['Channel']}</span>
                </div>
            </td>
            <td style='{cell_style}'>{_type_badge(row['Type'])}</td>
            <td style='{cell_style}'>{_shapley_bar(row['Shapley Wt (%)'])}</td>
            <td style='{cell_style};color:#555'>${row['Current Spend']:,.0f}</td>
            <td style='{cell_style};font-weight:600;color:#212529'>${row['New Spend']:,.0f}</td>
            <td style='{cell_style};font-weight:700;color:{sd_col}'>{sp_sign}${row['sp_delta_raw']:,.0f}</td>
            <td style='{cell_style};color:#555'>{row['Curr Conv (est)']:.1f}</td>
            <td style='{cell_style};color:#212529'>{row['New Conv (est)']:.1f}</td>
            <td style='{cell_style};font-weight:700;color:{lf_col}'>{lf_sign}{row['lift_raw']:.1f}</td>
            <td style='{cell_style};background:{mr_bg};font-weight:700;
                        color:{mr_fg};border-radius:6px;text-align:center'>
                {row['mroi_raw']:.2f}x
            </td>
        </tr>"""

    table_html = f"""
    <div style='overflow-x:auto;border-radius:12px;
                border:1px solid #dee2e6;margin-top:8px;box-shadow:0 1px 4px rgba(0,0,0,.06)'>
        <table style='width:100%;border-collapse:collapse;font-family:Segoe UI,sans-serif'>
            <thead>
                <tr>
                    {''.join(f"<th style='{header_style}'>{h}</th>" for h in headers)}
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    # ── legend ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='display:flex;gap:20px;flex-wrap:wrap;margin-top:10px;
                font-size:0.75rem;color:#6c757d'>
        <span><span style='display:inline-block;width:10px;height:10px;
              border-radius:2px;background:#d5f5e3;border:1px solid #a9dfbf;
              margin-right:4px'></span>Marginal ROI > 1.5x — high efficiency</span>
        <span><span style='display:inline-block;width:10px;height:10px;
              border-radius:2px;background:#eafaf1;border:1px solid #a9dfbf;
              margin-right:4px'></span>1.0x – 1.5x — above break-even</span>
        <span><span style='display:inline-block;width:10px;height:10px;
              border-radius:2px;background:#fde8e8;border:1px solid #f5b7b1;
              margin-right:4px'></span>Negative — spend cut, conversions lost</span>
        <span style='margin-left:auto'>Sorted by Conv Lift ↓</span>
    </div>
    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Scenario Comparison
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 💾 Section 4 — Scenario Comparison")
    st.markdown(
        "Save your current scenario and compare up to **4 named scenarios** side-by-side. "
        "Useful for stakeholder presentations — 'Scenario A is the base, B is growth, C is the cut.'"
    )
    st.markdown(
        "> 💡 **Tip:** Run the tab with different budgets or allocation modes, "
        "save each as a named scenario, then read the comparison table to decide which to present."
    )

    if "sc_saved" not in st.session_state:
        st.session_state["sc_saved"] = {}

    sc_name = st.text_input("Scenario name", value=f"Scenario {len(st.session_state['sc_saved'])+1}")

    if st.button("💾 Save this scenario"):
        if len(st.session_state["sc_saved"]) >= 4:
            st.warning("Maximum 4 scenarios saved. Clear one to add more.")
        else:
            st.session_state["sc_saved"][sc_name] = {
                "budget":      sc_new_budget,
                "delta":       sc_delta,
                "mode":        alloc_mode,
                "proj_conv":   round(proj_new_conv, 1),
                "lift_conv":   round(proj_lift_conv, 1),
                "lift_pct":    round(proj_lift_pct, 2),
                "ci_lower":    round(ci_lower_conv, 1),
                "ci_upper":    round(ci_upper_conv, 1),
                "significant": is_significant,
                "spend_snap":  {CHANNEL_LABELS[ch]: round(sc_new_spend.get(ch, 0), 0) for ch in CHANNELS},
            }
            st.success(f"✅ Saved: **{sc_name}**")

    if st.session_state["sc_saved"]:
        if st.button("🗑️ Clear all scenarios"):
            st.session_state["sc_saved"] = {}
            st.rerun()

        st.markdown("#### Saved Scenarios")
        comp_rows = []
        for sname, sdata in st.session_state["sc_saved"].items():
            comp_rows.append({
                "Scenario":        sname,
                "Mode":            sdata["mode"].replace("🤖 ", "").replace("🎛️ ", ""),
                "New Budget ($)":  sdata["budget"],
                "Δ Budget ($)":    sdata["delta"],
                "Proj. Conv.":     sdata["proj_conv"],
                "Conv. Lift":      sdata["lift_conv"],
                "Lift (%)":        sdata["lift_pct"],
                "CI Lower":        sdata["ci_lower"],
                "CI Upper":        sdata["ci_upper"],
                "Significant?":    "✅ Yes" if sdata["significant"] else "⚠️ No",
            })

        comp_df = pd.DataFrame(comp_rows)
        st.dataframe(
            comp_df.style.format({
                "New Budget ($)":  "${:,.0f}",
                "Δ Budget ($)":    "${:+,.0f}",
                "Proj. Conv.":     "{:,.1f}",
                "Conv. Lift":      "{:+.1f}",
                "Lift (%)":        "{:+.2f}%",
                "CI Lower":        "{:+.1f}",
                "CI Upper":        "{:+.1f}",
            }).background_gradient(subset=["Lift (%)"], cmap="RdYlGn"),
            width='stretch', hide_index=True,
        )

        # Visual comparison bar chart
        if len(comp_rows) > 1:
            fig_comp_sc = go.Figure()
            for row in comp_rows:
                fig_comp_sc.add_trace(go.Bar(
                    name=row["Scenario"],
                    x=["Projected Conversions", "Conversion Lift"],
                    y=[row["Proj. Conv."], row["Conv. Lift"]],
                    text=[f"{row['Proj. Conv.']:,.1f}", f"{row['Conv. Lift']:+.1f}"],
                    textposition="outside",
                ))
            fig_comp_sc.update_layout(
                barmode="group", height=380, template="plotly_white",
                title="Scenario Comparison — Projected Conversions & Lift",
                yaxis_title="Conversions",
                legend=dict(orientation="h", y=1.1),
                margin=dict(t=70, b=50),
            )
            st.plotly_chart(fig_comp_sc, width='stretch')

            # Spend breakdown per scenario
            st.markdown("#### Channel Spend Breakdown by Scenario")
            spend_rows = []
            for sname, sdata in st.session_state["sc_saved"].items():
                for ch_lbl, sp in sdata["spend_snap"].items():
                    spend_rows.append({"Scenario": sname, "Channel": ch_lbl, "Spend ($)": sp})
            spend_comp_df = pd.DataFrame(spend_rows)
            fig_spend_comp = px.bar(
                spend_comp_df, x="Channel", y="Spend ($)", color="Scenario",
                barmode="group", height=400, template="plotly_white",
                title="Per-Channel Spend by Scenario",
            )
            fig_spend_comp.update_xaxes(tickangle=-25)
            st.plotly_chart(fig_spend_comp, width='stretch')
    else:
        st.info("No scenarios saved yet. Configure a scenario above and click **Save**.")

    # ── Methodological note ───────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📚 Methodology & Limitations"):
        st.markdown(f"""
        **Response Curve Model**

        Each channel's conversion response to spend follows:
        """)
        st.latex(r"R_i(\text{spend}) = \alpha_i \cdot \text{spend}^{0.5}")
        st.markdown(f"""
        where $\\alpha_i$ is proportional to the channel's Shapley attribution weight.
        This is a stylised diminishing-returns model — doubling spend yields ~1.41× conversions,
        not 2×. Real saturation curves should be fitted from MMM historical data
        (see the **MMM + MTA Hybrid** tab for the path to production-grade curves).

        **Confidence Bands**

        Uncertainty in Shapley weights (from bootstrap resampling) is propagated through the
        response curve via a first-order linear approximation:

        $\\sigma_{{\\text{{lift}}}}^2 = \\sum_i \\left(\\frac{{\\partial R_i}}{{\\partial \\alpha_i}} \\cdot \\sigma_{{\\alpha_i}}\\right)^2$

        With Bootstrap CIs **{"enabled (n=" + str(n_bootstrap) + " resamples)" if run_ci else "disabled"}**,
        bands use {"actual bootstrap standard errors" if run_ci else "a fallback ±15% of each channel's Shapley weight (conservative estimate)"}.
        Enable Bootstrap CIs in the sidebar for tighter, data-driven bands.

        **Statistical Significance**

        Significance is assessed via a two-sided z-test:
        $z = \\text{{lift}} / \\sigma_{{\\text{{lift}}}}$. The threshold is $|z| > 1.96$ (95% confidence).
        This is an approximation — in production, use a proper experiment (A/B geo-test or
        synthetic control) to validate lift.

        **MTA vs MMM caveat**

        This simulator uses MTA Shapley weights as attribution signals.
        MTA cannot see offline channels (TV, Radio, Direct Mail, Agent Visit) — they receive
        near-zero Shapley credit. Offline spend changes are therefore **underestimated** here.
        For offline budget scenarios, use the MMM contribution shares from the Hybrid tab as
        the response curve alphas instead.
        """)


st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#95a5a6; font-size:0.8rem; padding:1rem 0">
    Multi-Touch Attribution Demo · Cooperative Game Theory (Shapley, Banzhaf, Markov) ·
    GBT Characteristic Function · Plackett-Luce Ordered Shapley · Bootstrap CIs ·
    Built with Streamlit &amp; Plotly ·
    <em>Zhao et al. (2018) · Grabisch &amp; Roubens (1999) · Anderl et al. (2016)</em>
</div>
""", unsafe_allow_html=True)