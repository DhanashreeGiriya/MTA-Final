"""
Plotly visualisation helpers for the MTA demo.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .data_generator import CHANNEL_COLORS, CHANNEL_LABELS, CHANNEL_TYPE, CHANNELS


# ── Colour helpers ────────────────────────────────────────────────────────────

def _ch_color(ch: str) -> str:
    return CHANNEL_COLORS.get(ch, "#999999")


MODEL_COLORS = {
    "Last Touch":       "#e74c3c",
    "First Touch":      "#e67e22",
    "Linear":           "#f39c12",
    "Time Decay":       "#2ecc71",
    "Position-Based":   "#1abc9c",
    "Markov Chain":     "#3498db",
    "Shapley":          "#9b59b6",
    "Shapley (Ordered)":"#8e44ad",
    "Banzhaf":          "#2980b9",
}


# ── 1. Attribution comparison bar chart ───────────────────────────────────────

def attribution_comparison(attr_df: pd.DataFrame, selected_models: List[str]) -> go.Figure:
    """
    Grouped bar chart: channels (x) × models (colour groups).
    attr_df has channels as index, models as columns (fractional credit).
    """
    df = attr_df[selected_models].copy()
    ch_labels = df.index.tolist()
    fig = go.Figure()
    for model in selected_models:
        fig.add_trace(go.Bar(
            name=model,
            x=ch_labels,
            y=(df[model] * 100).round(2),
            marker_color=MODEL_COLORS.get(model, "#aaa"),
            text=(df[model] * 100).round(1).astype(str) + "%",
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group",
        title="Attribution Credit Comparison (%) by Channel & Model",
        xaxis_title="Channel",
        yaxis_title="Attribution Credit (%)",
        legend_title="Model",
        height=480,
        template="plotly_white",
        font=dict(size=12),
        margin=dict(t=60, b=60),
    )
    fig.update_xaxes(tickangle=-30)
    return fig


# ── 2. Shapley waterfall ──────────────────────────────────────────────────────

def shapley_waterfall(shapley_values: Dict[str, float]) -> go.Figure:
    """Waterfall chart showing each channel's marginal Shapley contribution."""
    items = sorted(shapley_values.items(), key=lambda x: -x[1])
    labels = [CHANNEL_LABELS.get(k, k) for k, _ in items]
    values = [v * 100 for _, v in items]
    colors = [_ch_color(k) for k, _ in items]

    cumulative = np.cumsum([0] + values[:-1])
    fig = go.Figure()
    for i, (label, val, base, color) in enumerate(zip(labels, values, cumulative, colors)):
        fig.add_trace(go.Bar(
            name=label,
            x=[label],
            y=[val],
            base=[base],
            marker_color=color,
            text=f"{val:.1f}%",
            textposition="inside",
            showlegend=False,
        ))

    fig.update_layout(
        title="Shapley Values — Marginal Contribution per Channel",
        yaxis_title="Cumulative Attribution Credit (%)",
        height=420,
        template="plotly_white",
        barmode="stack",
        margin=dict(t=60, b=60),
    )
    fig.update_xaxes(tickangle=-20)
    return fig


# ── 3. Radar chart — multi-model comparison for a single channel ──────────────

def model_radar(attr_df: pd.DataFrame, channel_label: str) -> go.Figure:
    """Spider chart showing how different models credit the same channel."""
    if channel_label not in attr_df.index:
        return go.Figure()
    row = attr_df.loc[channel_label] * 100
    models = row.index.tolist()
    values = row.values.tolist()
    values_closed = values + [values[0]]
    models_closed = models + [models[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=models_closed,
        fill="toself",
        fillcolor="rgba(155,89,182,0.25)",
        line=dict(color="#9b59b6", width=2),
        name=channel_label,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(values) * 1.2])),
        title=f"Model Sensitivity — {channel_label}",
        height=400,
        template="plotly_white",
    )
    return fig


# ── 4. Channel interaction heatmap ────────────────────────────────────────────

def interaction_heatmap(interaction_df: pd.DataFrame) -> go.Figure:
    """
    Heatmap of pairwise Shapley Interaction Index.
    Red = synergy, Blue = substitution.
    """
    z = interaction_df.values
    labels = interaction_df.columns.tolist()

    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale=[
            [0.0, "#2166ac"],
            [0.5, "#f7f7f7"],
            [1.0, "#d6604d"],
        ],
        zmid=0,
        text=np.round(z, 4),
        texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(title="Interaction<br>Index"),
    ))
    fig.update_layout(
        title="Shapley Pairwise Interaction Index<br><sub>Red = synergy · Blue = substitution</sub>",
        height=500,
        template="plotly_white",
        xaxis=dict(tickangle=-40),
        margin=dict(t=80, b=80, l=100, r=40),
    )
    return fig


# ── 5. Sankey journey diagram ─────────────────────────────────────────────────

def journey_sankey(journeys: List[Dict], top_n: int = 200) -> go.Figure:
    """
    Sankey diagram of channel-to-channel transitions.
    Nodes coloured by channel type (online=blue, offline=orange).
    """
    from collections import defaultdict

    transition_counts: Dict[tuple, int] = defaultdict(int)
    conv_journeys = [j for j in journeys if j["converted"]][:top_n]

    for j in conv_journeys:
        path = j["path"]
        if len(path) < 2:
            continue
        for a, b in zip(path, path[1:]):
            transition_counts[(a, b)] += 1
        # Final → conversion
        transition_counts[(path[-1], "__conv__")] += 1

    all_nodes = []
    seen = set()
    for (a, b) in transition_counts:
        for n in (a, b):
            if n not in seen:
                all_nodes.append(n)
                seen.add(n)
    node_idx = {n: i for i, n in enumerate(all_nodes)}

    def _node_color(n: str) -> str:
        if n == "__conv__":
            return "#27ae60"
        return "rgba(31,119,180,0.8)" if CHANNEL_TYPE.get(n, "Online") == "Online" else "rgba(214,95,34,0.8)"

    def _node_label(n: str) -> str:
        return "✓ Conversion" if n == "__conv__" else CHANNEL_LABELS.get(n, n)

    node_colors  = [_node_color(n) for n in all_nodes]
    node_labels  = [_node_label(n) for n in all_nodes]

    sources, targets, values = [], [], []
    for (a, b), cnt in transition_counts.items():
        sources.append(node_idx[a])
        targets.append(node_idx[b])
        values.append(cnt)

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=20, thickness=20,
            label=node_labels,
            color=node_colors,
            line=dict(color="white", width=0.5),
        ),
        link=dict(source=sources, target=targets, value=values,
                  color="rgba(180,180,180,0.35)"),
    ))
    fig.update_layout(
        title="Channel-to-Channel Flow (Converting Journeys)",
        height=520,
        template="plotly_white",
        margin=dict(t=70, b=20, l=20, r=20),
    )
    return fig


# ── 6. Budget optimizer waterfall ─────────────────────────────────────────────

def budget_waterfall(opt_df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing current vs optimised spend."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Current Spend",
        x=opt_df["channel_label"],
        y=opt_df["current_spend"],
        marker_color="#95a5a6",
    ))
    fig.add_trace(go.Bar(
        name="Optimised Spend",
        x=opt_df["channel_label"],
        y=opt_df["optimised_spend"],
        marker_color="#9b59b6",
    ))
    fig.update_layout(
        barmode="group",
        title="Budget Reallocation: Current vs Shapley-Optimised",
        yaxis_title="Budget (USD)",
        xaxis_title="Channel",
        height=430,
        template="plotly_white",
        legend_title="Scenario",
        margin=dict(t=60),
    )
    fig.update_xaxes(tickangle=-25)
    return fig


def budget_delta_chart(opt_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of delta (increase/decrease)."""
    df = opt_df.sort_values("delta")
    colors = ["#e74c3c" if d < 0 else "#27ae60" for d in df["delta"]]
    fig = go.Figure(go.Bar(
        x=df["delta"],
        y=df["channel_label"],
        orientation="h",
        marker_color=colors,
        text=["${:+,.0f}".format(d) for d in df["delta"]],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_width=1.5, line_color="#333")
    fig.update_layout(
        title="Budget Delta per Channel (Optimised − Current)",
        xaxis_title="Change (USD)",
        height=400,
        template="plotly_white",
        margin=dict(t=60, l=120),
    )
    return fig


# ── 7. Markov transition heatmap ──────────────────────────────────────────────

def markov_transition_heatmap(journeys: List[Dict]) -> go.Figure:
    """Heatmap of transition probabilities between channels."""
    from collections import defaultdict
    ch_list = CHANNELS
    labels  = [CHANNEL_LABELS[c] for c in ch_list]
    idx     = {c: i for i, c in enumerate(ch_list)}
    n = len(ch_list)
    mat = np.zeros((n, n))

    for j in journeys:
        path = j["path"]
        for a, b in zip(path, path[1:]):
            if a in idx and b in idx:
                mat[idx[a], idx[b]] += 1

    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    mat = mat / row_sums

    fig = go.Figure(go.Heatmap(
        z=mat,
        x=labels,
        y=labels,
        colorscale="Blues",
        text=np.round(mat, 3),
        texttemplate="%{text}",
        textfont=dict(size=8),
        colorbar=dict(title="Transition<br>Probability"),
    ))
    fig.update_layout(
        title="Markov Transition Probabilities (From → To)",
        height=480,
        template="plotly_white",
        xaxis=dict(tickangle=-40, title="To Channel"),
        yaxis=dict(title="From Channel"),
        margin=dict(t=70, b=90, l=120, r=40),
    )
    return fig


# ── 8. Funnel overview KPIs ───────────────────────────────────────────────────

def channel_funnel_bar(summary_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar: touchpoints vs conversions per channel."""
    df = summary_df.sort_values("touchpoints", ascending=True)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Touchpoints", "Conversions"))
    colors = [_ch_color(ch) for ch in df["channel"]]

    fig.add_trace(go.Bar(
        y=df["channel_label"], x=df["touchpoints"],
        orientation="h", marker_color=colors, showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        y=df["channel_label"], x=df["conversions"],
        orientation="h", marker_color=colors, showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        height=420, template="plotly_white",
        title="Channel Volume: Touchpoints vs Conversions",
        margin=dict(t=70, l=120),
    )
    return fig


def conversion_rate_bar(summary_df: pd.DataFrame) -> go.Figure:
    """Bar chart: conversion rate per channel."""
    df = summary_df.sort_values("conv_rate", ascending=False)
    colors = [_ch_color(ch) for ch in df["channel"]]
    fig = go.Figure(go.Bar(
        x=df["channel_label"],
        y=(df["conv_rate"] * 100).round(1),
        marker_color=colors,
        text=(df["conv_rate"] * 100).round(1).astype(str) + "%",
        textposition="outside",
    ))
    fig.update_layout(
        title="Conversion Rate by Channel",
        yaxis_title="Conversion Rate (%)",
        height=380,
        template="plotly_white",
        margin=dict(t=60),
    )
    fig.update_xaxes(tickangle=-25)
    return fig


# ── 9. Shapley Bootstrap CI error-bar chart ───────────────────────────────────

def shapley_ci_chart(ci_df: "pd.DataFrame") -> "go.Figure":
    """
    Horizontal bar chart with 95% CI error bars for Shapley values.

    Each bar is the GBT point estimate; whiskers span [lower_ci, upper_ci].
    Channels sorted by point_estimate (largest first).

    Parameters
    ----------
    ci_df : output of shapley_bootstrap_ci() — must have columns
            channel_label, point_estimate, lower_ci, upper_ci.
    """
    df = ci_df.sort_values("point_estimate", ascending=True)

    # Asymmetric error bars (Plotly expects [lower_error, upper_error])
    err_minus = ((df["point_estimate"] - df["lower_ci"]) * 100).clip(lower=0)
    err_plus  = ((df["upper_ci"] - df["point_estimate"]) * 100).clip(lower=0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["channel_label"],
        x=(df["point_estimate"] * 100).round(2),
        orientation="h",
        marker_color="#9b59b6",
        name="Point estimate",
        error_x=dict(
            type="data",
            arrayminus=err_minus.tolist(),
            array=err_plus.tolist(),
            color="#4a235a",
            thickness=2.5,
            width=6,
        ),
        text=(df["point_estimate"] * 100).round(1).astype(str) + "%",
        textposition="outside",
    ))

    # Annotate asymmetric CI bounds to the right of each bar
    for _, row in df.iterrows():
        lo_pct = row["lower_ci"] * 100
        hi_pct = row["upper_ci"] * 100
        fig.add_annotation(
            y=row["channel_label"],
            x=(hi_pct) + 0.4,
            text=f"[{lo_pct:.1f}%, {hi_pct:.1f}%]",
            showarrow=False,
            font=dict(size=8, color="#7f8c8d"),
            xanchor="left",
        )

    fig.update_layout(
        title="Shapley Values with 95% Bootstrap Confidence Intervals",
        xaxis_title="Attribution Credit (%)",
        height=460,
        template="plotly_white",
        margin=dict(t=60, l=130, r=120),
        showlegend=False,
    )
    return fig
