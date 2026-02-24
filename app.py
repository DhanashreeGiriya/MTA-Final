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
             use_container_width=True)
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


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎯 Multi-Touch Attribution Demo")
st.markdown(
    "**Cooperative Game Theory–powered attribution** — Shapley values, Banzhaf index, "
    "Ordered Shapley (Zhao 2018), and Markov chain removal effects, compared against "
    "classic heuristic baselines across **10 channels** (6 online + 4 offline)."
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_compare, tab_shapley, tab_synergy, tab_journey, tab_budget, tab_markov = st.tabs([
    "📊 Overview",
    "🎯 Model Comparison",
    "🔬 Shapley Deep Dive",
    "🔗 Channel Synergies",
    "🛤️ Journey Explorer",
    "💰 Budget Optimizer",
    "📈 Markov Analysis",
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
        st.plotly_chart(channel_funnel_bar(summary_df), use_container_width=True)
    with col_right:
        st.plotly_chart(conversion_rate_bar(summary_df), use_container_width=True)

    st.markdown("---")
    st.subheader("🧾 Channel Summary")
    disp = summary_df[["channel_label","channel_type","touchpoints","conversions","conv_rate","revenue"]].copy()
    disp.columns = ["Channel","Type","Touchpoints","Conversions","Conv. Rate","Revenue ($)"]
    disp["Conv. Rate"] = (disp["Conv. Rate"] * 100).round(1).astype(str) + "%"
    disp["Revenue ($)"] = disp["Revenue ($)"].apply(lambda x: f"${x:,.0f}")
    st.dataframe(disp, use_container_width=True, hide_index=True)

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
                 use_container_width=True, hide_index=True)


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
        st.plotly_chart(attribution_comparison(attr_df, selected), use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Attribution Matrix (% credit)")
        pct_df = (attr_df[selected] * 100).round(2)
        st.dataframe(
            pct_df.style.background_gradient(cmap="Purples", axis=None),
            use_container_width=True,
        )

        st.markdown("---")
        st.subheader("🔍 Channel-Level Model Sensitivity")
        ch_label = st.selectbox("Select channel", attr_df.index.tolist())
        st.plotly_chart(model_radar(attr_df[selected], ch_label), use_container_width=True)

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
            st.plotly_chart(shapley_waterfall(sh_vals), use_container_width=True)
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
        st.dataframe(sub_df.style.background_gradient(cmap="Purples"), use_container_width=True)

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
        st.plotly_chart(fig_diff, use_container_width=True)
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

        st.plotly_chart(shapley_ci_chart(ci_df), use_container_width=True)

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
        st.dataframe(ci_disp, use_container_width=True, hide_index=True)
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

    st.plotly_chart(interaction_heatmap(int_df), use_container_width=True)

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
        top5 = synergy_df.head(5)
        top5["Relationship"] = "Synergy 🟢"
        st.dataframe(top5, use_container_width=True, hide_index=True)
    with col_sub:
        st.markdown("**Top 5 Substitutable Pairs** ↔️")
        bot5 = synergy_df.tail(5).sort_values("Index")
        bot5["Relationship"] = "Substitution 🔴"
        st.dataframe(bot5, use_container_width=True, hide_index=True)

    st.info("💡 Use synergistic pairs to inform **media mix** decisions — channels that amplify "
            "each other's impact should be run concurrently, not traded off against each other.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — JOURNEY EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_journey:
    st.subheader("🛤️ Customer Journey Flow")

    sankey_n = st.slider("Number of converting journeys to visualise", 50, 500, 200, step=50)
    st.plotly_chart(journey_sankey(journeys, top_n=sankey_n), use_container_width=True)

    st.markdown("---")
    col_paths, col_stats = st.columns([3, 2])

    with col_paths:
        st.subheader("🏆 Top Converting Paths")
        st.dataframe(paths_df, use_container_width=True, hide_index=True)

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
        st.plotly_chart(fig_len, use_container_width=True)

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
    st.plotly_chart(fig_mix, use_container_width=True)


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
            st.plotly_chart(budget_waterfall(opt_df), use_container_width=True)
        with col_r:
            st.plotly_chart(budget_delta_chart(opt_df), use_container_width=True)

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
        st.dataframe(disp_opt, use_container_width=True, hide_index=True)

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
        st.plotly_chart(markov_transition_heatmap(journeys), use_container_width=True)

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
        st.plotly_chart(fig_mkv, use_container_width=True)

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
        st.plotly_chart(fig_comp, use_container_width=True)

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


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#95a5a6; font-size:0.8rem; padding:1rem 0">
    Multi-Touch Attribution Demo · Cooperative Game Theory (Shapley, Banzhaf, Markov) ·
    GBT Characteristic Function · Plackett-Luce Ordered Shapley · Bootstrap CIs ·
    Built with Streamlit &amp; Plotly ·
    <em>Zhao et al. (2018) · Grabisch &amp; Roubens (1999) · Anderl et al. (2016)</em>
</div>
""", unsafe_allow_html=True)
