#!/usr/bin/env python3
"""
Streamlit analysis app for bulk valley cross-section exports.

Usage
-----
    streamlit run cross_section/analyze_valley_sections.py

Load the CSV produced by the Bulk Export tab of mark_valley_line.py,
then explore the data across five analysis views:

    Overview    – map scatter + summary statistics
    Direction   – compass-sector comparison (histogram + violin)
    Along Valley– metric evolution along each valley line
    Scatter     – 2-D metric vs. metric, coloured by any grouping
    Correlation – pairwise Pearson/Spearman heatmap
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── constants ────────────────────────────────────────────────────────────────

_COMPASS16 = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
              "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
_CARDINAL_COLORS = ["#4ecdc4", "#ff6b35", "#ffe66d", "#c77dff"]   # N, E, S, W

METRIC_LABELS: dict[str, str] = {
    "total_WD":            "Total W/D",
    "left_W":              "Left Width (m)",
    "right_W":             "Right Width (m)",
    "left_D":              "Left Depth (m)",
    "right_D":             "Right Depth (m)",
    "left_WD100":          "Left W/D @100%D",
    "right_WD100":         "Right W/D @100%D",
    "left_WD10":           "Left W/D @10%D",
    "right_WD10":          "Right W/D @10%D",
    "left_b":              "Left shape exp. b",
    "right_b":             "Right shape exp. b",
    "left_r2":             "Left fit R²",
    "right_r2":            "Right fit R²",
    "left_max_slope_deg":  "Left max slope (°)",
    "right_max_slope_deg": "Right max slope (°)",
    "left_mean_slope_deg": "Left mean slope (°)",
    "right_mean_slope_deg":"Right mean slope (°)",
    "left_floor_width_m":  "Left floor width (m)",
    "right_floor_width_m": "Right floor width (m)",
    "left_slope_p90_deg":  "Left slope @90%h (°)",
    "right_slope_p90_deg": "Right slope @90%h (°)",
    "left_slope_p70_deg":  "Left slope @70%h (°)",
    "right_slope_p70_deg": "Right slope @70%h (°)",
    "left_slope_p50_deg":  "Left slope @50%h (°)",
    "right_slope_p50_deg": "Right slope @50%h (°)",
    "left_slope_p30_deg":  "Left slope @30%h (°)",
    "right_slope_p30_deg": "Right slope @30%h (°)",
    "left_slope_p10_deg":  "Left slope @10%h (°)",
    "right_slope_p10_deg": "Right slope @10%h (°)",
    "diff_W":              "Diff width L−R (m)",
    "diff_D":              "Diff depth L−R (m)",
    "diff_WD100":          "Diff W/D@100%D L−R",
    "diff_WD10":           "Diff W/D@10%D L−R",
    "diff_b":              "Diff shape exp. b L−R",
    "diff_r2":             "Diff fit R² L−R",
    "diff_mean_slope_deg": "Diff mean slope L−R (°)",
    "diff_max_slope_deg":  "Diff max slope L−R (°)",
    "diff_floor_width_m":  "Diff floor width L−R (m)",
    "diff_slope_p90_deg":  "Diff slope @90%h L−R (°)",
    "diff_slope_p70_deg":  "Diff slope @70%h L−R (°)",
    "diff_slope_p50_deg":  "Diff slope @50%h L−R (°)",
    "diff_slope_p30_deg":  "Diff slope @30%h L−R (°)",
    "diff_slope_p10_deg":  "Diff slope @10%h L−R (°)",
    "left_dist_h50_m":     "Left dist @50% lower D (m)",
    "right_dist_h50_m":    "Right dist @50% lower D (m)",
    "diff_dist_h50_m":     "Diff dist @50% lower D L−R (m)",
    "left_dist_h90_m":     "Left dist @90% lower D (m)",
    "right_dist_h90_m":    "Right dist @90% lower D (m)",
    "diff_dist_h90_m":     "Diff dist @90% lower D L−R (m)",
    "A_W":                 "Width asymmetry Aᵂ",
    "A_S":                 "Slope asymmetry Aₛ (°)",
    "A_b":                 "Shape asymmetry Aᵇ",
    "skewness":            "Profile skewness",
    "valley_pct":          "Position along valley (%)",
    "bottom_elev_m":       "Valley floor elevation (m)",
    "valley_bearing_deg":  "Valley bearing (°)",
    "cs_bearing_deg":      "Cross-section bearing (°)",
    "left_bearing_deg":    "Left flank aspect (°)",
    "right_bearing_deg":   "Right flank aspect (°)",
    "line_length_m":       "Line length (m)",
}

FLANK_METRIC_LABELS: dict[str, str] = {
    "W":             "Width (m)",
    "D":             "Depth (m)",
    "WD100":         "W/D @100%D",
    "WD10":          "W/D @10%D",
    "b":             "Shape exp. b",
    "r2":            "Fit R\u00b2",
    "max_slope_deg": "Max slope (\u00b0)",
    "mean_slope_deg":"Mean slope (\u00b0)",
    "floor_width_m": "Floor width (m)",
    "slope_p90_deg": "Slope @90%h (\u00b0)",
    "slope_p70_deg": "Slope @70%h (\u00b0)",
    "slope_p50_deg": "Slope @50%h (\u00b0)",
    "slope_p30_deg": "Slope @30%h (\u00b0)",
    "slope_p10_deg": "Slope @10%h (\u00b0)",
    "dist_h50_m":    "Dist @50% lower D (m)",
    "dist_h90_m":    "Dist @90% lower D (m)",
}
FLANK_METRICS = list(FLANK_METRIC_LABELS.keys())

FILTER_METRICS = [
    "total_WD", "left_W", "right_W", "left_D", "right_D",
    "left_b", "right_b", "left_r2", "right_r2",
    "left_max_slope_deg", "right_max_slope_deg",
    "A_W", "A_S", "A_b", "skewness", "bottom_elev_m",
]

ANALYSIS_METRICS = [k for k in METRIC_LABELS if k not in (
    "valley_bearing_deg", "cs_bearing_deg",
    "left_bearing_deg", "right_bearing_deg",
    "valley_pct", "line_length_m",
)]


# ── helpers ──────────────────────────────────────────────────────────────────

def _bin_label(center: float) -> str:
    """Label a direction bin by its centre bearing, e.g. ``N (000°)``."""
    deg = (f"{int(round(center)):03d}" if abs(center - round(center)) < 1e-6
           else f"{center:05.1f}")
    k = center / 22.5
    if abs(k - round(k)) < 1e-6:
        return f"{_COMPASS16[int(round(k)) % 16]} ({deg}°)"
    return f"{deg}°"


def direction_scheme(n_bins: int) -> tuple[list[str], dict[str, str]]:
    """Ordered bin labels (centred on 0°, width 360/n) and a colour per label."""
    width = 360.0 / n_bins
    order = [_bin_label(i * width) for i in range(n_bins)]
    if n_bins == 4:
        colors = dict(zip(order, _CARDINAL_COLORS))
    else:
        import colorsys
        colors = {
            lbl: "#%02x%02x%02x" % tuple(
                int(255 * c) for c in colorsys.hsv_to_rgb(i / n_bins, 0.65, 0.95))
            for i, lbl in enumerate(order)
        }
    return order, colors


def classify_bearing(
    series: pd.Series, col: str = "valley_bearing_deg", n_bins: int = 4
) -> pd.Series:
    """Classify bearings into *n_bins* equal sectors centred on 0°, 360/n, …"""
    order, _ = direction_scheme(n_bins)
    width = 360.0 / n_bins
    b     = series % 360
    idx   = np.floor(((b + width / 2.0) % 360.0) / width)
    out   = pd.Series(np.nan, index=series.index, dtype=object)
    ok    = b.notna()
    out[ok] = [order[int(i) % n_bins] for i in idx[ok]]
    return out


def _label(col: str) -> str:
    return METRIC_LABELS.get(col, col)


def _available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _finite(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df[np.isfinite(df[col])]


def _melt_flanks(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback: build a flank-level dataframe from the section-level CSV.

    Prefers loading the pre-built ``_flanks.csv``; call this only when that
    file is not available.
    """
    FLANK_MAP = {k: (f"left_{k}", f"right_{k}") for k in FLANK_METRICS}
    PASS = ["section_id", "line_idx", "line_length_m", "section_in_line",
            "valley_pct", "bottom_elev_m", "bottom_lon", "bottom_lat",
            "valley_bearing_deg", "total_WD", "A_W", "A_S", "A_b", "skewness",
            *[f"diff_{k}" for k in FLANK_METRICS]]
    SIDE_META = [
        ("left",  "left_bearing_deg",  "left_lon",  "left_lat"),
        ("right", "right_bearing_deg", "right_lon", "right_lat"),
    ]
    halves = []
    for side, bk, lonk, latk in SIDE_META:
        half = df[[c for c in PASS if c in df.columns]].copy()
        half["side"]              = side
        half["flank_bearing_deg"] = df.get(bk, pd.Series(np.nan, index=df.index))
        half["flank_lon"]         = df.get(lonk, pd.Series(np.nan, index=df.index))
        half["flank_lat"]         = df.get(latk, pd.Series(np.nan, index=df.index))
        for metric, (lc, rc) in FLANK_MAP.items():
            col = lc if side == "left" else rc
            half[metric] = df.get(col, pd.Series(np.nan, index=df.index))
        halves.append(half)
    result = pd.concat(halves, ignore_index=True)
    return result.dropna(subset=["flank_bearing_deg"])



def sidebar_load() -> tuple["pd.DataFrame | None", "pd.DataFrame | None"]:
    st.sidebar.header("Data")
    default = str(Path(__file__).parent / "export" / "valley_sections.csv")
    csv_path = st.sidebar.text_input("CSV path", default)

    if not Path(csv_path).exists():
        st.sidebar.warning("File not found.")
        return None, None

    df = pd.read_csv(csv_path)
    st.sidebar.success(f"{len(df):,} sections loaded")

    # Auto-detect flanks CSV (same dir, stem + _flanks.csv)
    p = Path(csv_path)
    flanks_path = p.parent / (p.stem + "_flanks.csv")
    if flanks_path.exists():
        df_flanks = pd.read_csv(flanks_path)
        st.sidebar.success(f"{len(df_flanks):,} flanks loaded")
    else:
        df_flanks = None
        st.sidebar.info("No _flanks.csv found — will derive from sections.")

    return df, df_flanks


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    # Line selector
    if "line_idx" in df.columns and df["line_idx"].nunique() > 1:
        all_lines = sorted(df["line_idx"].unique())
        sel_lines = st.sidebar.multiselect(
            "Valley lines", all_lines,
            default=all_lines,
            format_func=lambda i: f"Line {int(i) + 1}",
        )
        if sel_lines:
            df = df[df["line_idx"].isin(sel_lines)]

    # Min R² quality gate
    min_r2 = st.sidebar.slider("Min fit R² (both flanks)", 0.0, 1.0, 0.0, 0.05)
    if min_r2 > 0 and "left_r2" in df.columns:
        df = df[
            (df["left_r2"].fillna(-1) >= min_r2) &
            (df["right_r2"].fillna(-1) >= min_r2)
        ]

    st.sidebar.markdown("**Metric ranges**")
    for col in _available(df, FILTER_METRICS):
        finite_vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if finite_vals.empty:
            continue
        lo, hi = float(finite_vals.min()), float(finite_vals.max())
        if lo >= hi:
            continue
        sel = st.sidebar.slider(
            _label(col), lo, hi, (lo, hi),
            key=f"filt_{col}",
            format="%.2f",
        )
        df = df[
            (df[col].fillna(lo - 1) >= sel[0]) &
            (df[col].fillna(hi + 1) <= sel[1])
        ]

    return df


# ── tab: overview ────────────────────────────────────────────────────────────

def tab_overview(df: pd.DataFrame) -> None:
    st.subheader("Overview")

    # Map scatter
    if {"center_lon", "center_lat"}.issubset(df.columns):
        color_col = st.selectbox(
            "Colour by", _available(df, ANALYSIS_METRICS),
            index=_available(df, ANALYSIS_METRICS).index("total_WD")
            if "total_WD" in _available(df, ANALYSIS_METRICS) else 0,
            key="ov_color",
        )
        df_map = df.dropna(subset=["center_lon", "center_lat", color_col])
        fig = px.scatter(
            df_map, x="center_lon", y="center_lat",
            color=color_col, color_continuous_scale="Viridis",
            labels={"center_lon": "Longitude", "center_lat": "Latitude",
                    color_col: _label(color_col)},
            title=f"Section locations coloured by {_label(color_col)}",
            hover_data={c: True for c in ["line_idx", "valley_pct", "total_WD"]
                        if c in df.columns},
        )
        fig.update_layout(height=480, yaxis_scaleanchor="x")
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

    # Summary statistics table
    st.markdown("#### Summary statistics")
    stat_cols = _available(df, ANALYSIS_METRICS)
    st.dataframe(df[stat_cols].describe().round(3), width="stretch")

    # Per-line section count
    if "line_idx" in df.columns:
        st.markdown("#### Sections per line")
        counts = (
            df.groupby("line_idx")
            .agg(n_sections=("section_id", "count"),
                 pct_detected=("total_WD", lambda x: 100 * x.notna().mean()))
            .reset_index()
        )
        counts["line_idx"] = counts["line_idx"].apply(lambda i: f"Line {int(i)+1}")
        st.dataframe(counts.rename(columns={
            "line_idx": "Line", "n_sections": "Sections",
            "pct_detected": "% shoulder detected",
        }), width="stretch", hide_index=True)


# ── tab: direction ───────────────────────────────────────────────────────────

def tab_direction(df: pd.DataFrame, n_bins: int) -> None:
    st.subheader("Direction Analysis")
    dir_order, dir_colors = direction_scheme(n_bins)
    st.caption(
        f"Classifies each section by bearing into {n_bins} equal sectors, each labelled by "
        "its centre direction. Use the controls to choose which bearing column drives the "
        "grouping and which metric to compare."
    )

    d_col1, d_col2, d_col3 = st.columns(3)
    with d_col1:
        bearing_col = st.selectbox(
            "Direction from",
            [c for c in ["valley_bearing_deg", "left_bearing_deg", "right_bearing_deg",
                          "cs_bearing_deg"] if c in df.columns],
            format_func=_label,
            key="dir_bearing",
        )
    with d_col2:
        metric = st.selectbox(
            "Metric to compare",
            _available(df, ANALYSIS_METRICS),
            format_func=_label,
            key="dir_metric",
        )
    with d_col3:
        chart_type = st.radio("Chart type", ["KDE", "Violin", "Box"],
                              horizontal=True, key="dir_chart")

    df2 = df.dropna(subset=[bearing_col, metric]).copy()
    df2["direction"] = classify_bearing(df2[bearing_col], n_bins=n_bins)
    df2 = df2[df2["direction"].isin(dir_order)]

    if df2.empty:
        st.warning("No data after filtering.")
        return

    label_m = _label(metric)

    if chart_type == "KDE":
        from scipy.stats import gaussian_kde as _gkde
        fig = go.Figure()
        _pool = np.concatenate([df2[df2["direction"] == d][metric].dropna().values
                                 for d in dir_order
                                 if len(df2[df2["direction"] == d][metric].dropna()) >= 3
                                 ] or [np.array([])])
        _x_hi = float(np.percentile(_pool, 99)) if len(_pool) else None
        for d in dir_order:
            vals = df2[df2["direction"] == d][metric].dropna().values
            if len(vals) < 3:
                continue
            n = len(vals)
            kde = _gkde(vals)
            x_end = min(float(vals.max()), _x_hi) if _x_hi is not None else float(vals.max())
            x_rng = np.linspace(float(vals.min()), x_end, 300)
            y_rng = kde(x_rng)
            c_hex = dir_colors[d].lstrip("#")
            r, g, b_ = int(c_hex[0:2], 16), int(c_hex[2:4], 16), int(c_hex[4:6], 16)
            fig.add_trace(go.Scatter(
                x=x_rng, y=y_rng,
                mode="lines",
                name=f"{d}  (n={n})",
                line=dict(color=dir_colors[d], width=2.5),
                fill="tozeroy",
                fillcolor=f"rgba({r},{g},{b_},0.18)",
            ))
        fig.update_layout(
            xaxis_title=label_m,
            yaxis_title="Density",
            title=f"{label_m} by valley direction",
            legend_title="Direction",
            height=420,
        )
        if _x_hi is not None:
            fig.update_xaxes(range=[float(_pool.min()), _x_hi])

    elif chart_type == "Violin":
        fig = go.Figure()
        for d in dir_order:
            sub = df2[df2["direction"] == d][metric]
            if sub.empty:
                continue
            fig.add_trace(go.Violin(
                y=sub, name=f"{d}  (n={len(sub)})",
                box_visible=True,
                meanline_visible=True,
                fillcolor=dir_colors[d],
                line_color=dir_colors[d],
                opacity=0.7,
            ))
        fig.update_layout(
            yaxis_title=label_m,
            title=f"{label_m} by valley direction",
            height=420,
        )

    else:  # Box
        fig = go.Figure()
        for d in dir_order:
            sub = df2[df2["direction"] == d][metric]
            if sub.empty:
                continue
            fig.add_trace(go.Box(
                y=sub, name=f"{d}  (n={len(sub)})",
                marker_color=dir_colors[d],
                boxmean="sd",
            ))
        fig.update_layout(
            yaxis_title=label_m,
            title=f"{label_m} by valley direction",
            height=420,
        )

    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

    # ── N/S vs E/W side-by-side ──────────────────────────────────────────
    def _make_pair_chart(dirs: list[str], title: str) -> go.Figure:
        """Build the same chart type but for a subset of directions."""
        if chart_type == "KDE":
            from scipy.stats import gaussian_kde as _gkde
            f = go.Figure()
            _pool = np.concatenate([df2[df2["direction"] == d][metric].dropna().values
                                     for d in dirs
                                     if len(df2[df2["direction"] == d][metric].dropna()) >= 3
                                     ] or [np.array([])])
            _x_hi = float(np.percentile(_pool, 99)) if len(_pool) else None
            for d in dirs:
                vals = df2[df2["direction"] == d][metric].dropna().values
                if len(vals) < 3:
                    continue
                n = len(vals)
                kde = _gkde(vals)
                x_end = min(float(vals.max()), _x_hi) if _x_hi is not None else float(vals.max())
                x_rng = np.linspace(float(vals.min()), x_end, 300)
                y_rng = kde(x_rng)
                c_hex = dir_colors[d].lstrip("#")
                r, g, b_ = int(c_hex[0:2], 16), int(c_hex[2:4], 16), int(c_hex[4:6], 16)
                f.add_trace(go.Scatter(
                    x=x_rng, y=y_rng,
                    mode="lines",
                    name=f"{d}  (n={n})",
                    line=dict(color=dir_colors[d], width=2.5),
                    fill="tozeroy",
                    fillcolor=f"rgba({r},{g},{b_},0.18)",
                ))
            f.update_layout(xaxis_title=label_m, yaxis_title="Density",
                            title=title, legend_title="Direction", height=360)
            if _x_hi is not None:
                f.update_xaxes(range=[float(_pool.min()), _x_hi])
        elif chart_type == "Violin":
            f = go.Figure()
            for d in dirs:
                sub = df2[df2["direction"] == d][metric]
                if sub.empty:
                    continue
                f.add_trace(go.Violin(
                    y=sub, name=f"{d}  (n={len(sub)})",
                    box_visible=True, meanline_visible=True,
                    fillcolor=dir_colors[d], line_color=dir_colors[d], opacity=0.7,
                ))
            f.update_layout(yaxis_title=label_m, title=title, height=360)
        else:
            f = go.Figure()
            for d in dirs:
                sub = df2[df2["direction"] == d][metric]
                if sub.empty:
                    continue
                f.add_trace(go.Box(
                    y=sub, name=f"{d}  (n={len(sub)})",
                    marker_color=dir_colors[d], boxmean="sd",
                ))
            f.update_layout(yaxis_title=label_m, title=title, height=360)
        return f

    if n_bins % 2 == 0:   # opposite bins only exist for an even bin count
        half  = n_bins // 2
        pairs = [(dir_order[i], dir_order[i + half]) for i in range(half)]
        for row in range(0, len(pairs), 2):
            for col, (a, b) in zip(st.columns(2), pairs[row:row + 2]):
                with col:
                    st.plotly_chart(
                        _make_pair_chart([a, b], f"{label_m}: {a} vs {b}"),
                        width="stretch", config={"displaylogo": False},
                        key=f"dir_pair_{a}",
                    )

    # ── compass rose: mean metric per quadrant ──────────────────────────
    st.markdown("#### Mean value per direction bin")
    stats_rows = []
    for d in dir_order:
        sub = df2[df2["direction"] == d][metric]
        if sub.empty:
            continue
        stats_rows.append({
            "Direction": d,
            "n": len(sub),
            "mean": round(sub.mean(), 3),
            "median": round(sub.median(), 3),
            "std": round(sub.std(), 3),
            "min": round(sub.min(), 3),
            "max": round(sub.max(), 3),
        })
    st.dataframe(pd.DataFrame(stats_rows), width="stretch", hide_index=True)

    # ── polar heatmap of metric vs. bearing ─────────────────────────────
    st.markdown("#### Metric vs. bearing (polar)")
    df2["bearing_bin"] = (df2[bearing_col] // 10 * 10).astype(int)
    polar_df = df2.groupby("bearing_bin")[metric].median().reset_index()
    fig_polar = go.Figure(go.Barpolar(
        r=polar_df[metric] - polar_df[metric].min() + 1e-6,
        theta=polar_df["bearing_bin"],
        width=10,
        marker_colorscale="Viridis",
        marker_color=polar_df[metric],
        hovertemplate=f"Bearing: %{{theta}}°<br>Median {label_m}: %{{customdata:.3f}}",
        customdata=polar_df[metric],
    ))
    fig_polar.update_layout(
        title=f"Median {label_m} by 10° bearing bin",
        polar=dict(angularaxis=dict(direction="clockwise", rotation=90)),
        height=400,
    )
    st.plotly_chart(fig_polar, width="stretch", config={"displaylogo": False})


# ── tab: along valley ────────────────────────────────────────────────────────

def tab_along_valley(df: pd.DataFrame) -> None:
    st.subheader("Along-Valley Profile")
    st.caption(
        "Shows how a metric varies along each valley line from start (0%) to end (100%). "
        "Useful for detecting systematic changes in morphology along the channel."
    )

    av_col1, av_col2, av_col3 = st.columns(3)
    with av_col1:
        metric = st.selectbox(
            "Metric", _available(df, ANALYSIS_METRICS),
            format_func=_label, key="av_metric",
        )
    with av_col2:
        smooth_pts = st.slider("Rolling mean window (sections)", 1, 50, 5, key="av_smooth")
    with av_col3:
        show_raw = st.checkbox("Show raw points", value=True, key="av_raw")

    df2 = df.dropna(subset=["valley_pct", metric]).copy()
    label_m = _label(metric)

    has_lines = "line_idx" in df2.columns and df2["line_idx"].nunique() > 1
    line_groups = sorted(df2["line_idx"].unique()) if has_lines else [None]
    colors_line = px.colors.qualitative.Plotly

    fig = go.Figure()
    for li, line_id in enumerate(line_groups):
        sub = df2[df2["line_idx"] == line_id].sort_values("valley_pct") \
              if line_id is not None else df2.sort_values("valley_pct")
        name = f"Line {int(line_id) + 1}" if line_id is not None else "All sections"
        clr  = colors_line[li % len(colors_line)]

        if show_raw:
            fig.add_trace(go.Scatter(
                x=sub["valley_pct"], y=sub[metric],
                mode="markers",
                marker=dict(size=4, color=clr, opacity=0.35),
                name=f"{name} (raw)",
                showlegend=True,
            ))

        if smooth_pts > 1 and len(sub) >= smooth_pts:
            rolled = sub[metric].rolling(smooth_pts, center=True, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=sub["valley_pct"], y=rolled,
                mode="lines",
                line=dict(width=2.5, color=clr),
                name=f"{name} (smooth)",
            ))

    fig.update_layout(
        xaxis_title="Position along valley (%)",
        yaxis_title=label_m,
        title=f"{label_m} along the valley",
        height=440,
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

    # Correlation with valley_pct
    if len(df2) > 5:
        rho = df2["valley_pct"].corr(df2[metric])
        st.caption(f"Pearson r with valley position: **{rho:.3f}**")


# ── tab: scatter ─────────────────────────────────────────────────────────────

def tab_scatter(df: pd.DataFrame, n_bins: int) -> None:
    st.subheader("Scatter Plot")
    dir_order, dir_colors = direction_scheme(n_bins)

    sc_col1, sc_col2, sc_col3, sc_col4 = st.columns(4)
    avail = _available(df, ANALYSIS_METRICS)
    with sc_col1:
        x_col = st.selectbox("X axis", avail, format_func=_label,
                             index=avail.index("left_max_slope_deg")
                             if "left_max_slope_deg" in avail else 0,
                             key="sc_x")
    with sc_col2:
        y_col = st.selectbox("Y axis", avail, format_func=_label,
                             index=avail.index("total_WD")
                             if "total_WD" in avail else 1,
                             key="sc_y")
    with sc_col3:
        color_opts = ["direction (valley)", "direction (left flank)",
                      "direction (right flank)", "line_idx"] + avail
        color_sel = st.selectbox("Colour by", color_opts, key="sc_color")
    with sc_col4:
        size_col = st.selectbox(
            "Size by (optional)",
            ["(none)"] + avail,
            key="sc_size",
        )

    df2 = df.dropna(subset=[x_col, y_col]).copy()

    # Resolve colour
    if color_sel.startswith("direction"):
        if "valley" in color_sel:
            bcol = "valley_bearing_deg"
        elif "left" in color_sel:
            bcol = "left_bearing_deg"
        else:
            bcol = "right_bearing_deg"
        if bcol in df2.columns:
            df2["_color"] = classify_bearing(df2[bcol], n_bins=n_bins)
            df2 = df2.dropna(subset=["_color"])
            color_map = dir_colors
            color_col = "_color"
        else:
            color_col = None
            color_map = None
    elif color_sel == "line_idx":
        df2["_color"] = df2["line_idx"].astype(str)
        color_col = "_color"
        color_map = None
    else:
        df2["_color"] = df2[color_sel]
        color_col = "_color"
        color_map = None

    size_col_used = size_col if size_col != "(none)" and size_col in df2.columns else None
    if size_col_used:
        df2 = df2.dropna(subset=[size_col_used])
        # Normalise size to [4, 18]
        sv = df2[size_col_used]
        df2["_sz"] = 4 + 14 * (sv - sv.min()) / max(sv.max() - sv.min(), 1e-9)

    fig = px.scatter(
        df2,
        x=x_col, y=y_col,
        color=color_col,
        color_discrete_map=color_map if isinstance(color_map, dict) else None,
        color_continuous_scale="Viridis" if color_map is None and color_col else None,
        size="_sz" if size_col_used else None,
        size_max=18,
        opacity=0.65,
        labels={x_col: _label(x_col), y_col: _label(y_col),
                "_color": color_sel},
        hover_data={c: True for c in ["valley_pct", "line_idx", "bottom_elev_m"]
                    if c in df2.columns},
    )
    # Manual OLS trendline (no statsmodels required)
    _xv = df2[x_col].values
    _yv = df2[y_col].values
    _ok = np.isfinite(_xv) & np.isfinite(_yv)
    if _ok.sum() > 2:
        _m, _b = np.polyfit(_xv[_ok], _yv[_ok], 1)
        _xs = np.linspace(_xv[_ok].min(), _xv[_ok].max(), 100)
        fig.add_trace(go.Scatter(
            x=_xs, y=_m * _xs + _b,
            mode="lines", line=dict(color="white", width=1.5, dash="dot"),
            name="trend", showlegend=False,
        ))
    fig.update_layout(height=500)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

    # Quick Pearson r
    if len(df2) > 5:
        r = df2[x_col].corr(df2[y_col])
        st.caption(f"Pearson r = **{r:.3f}**  (n = {len(df2):,})")


# ── tab: correlation ─────────────────────────────────────────────────────────

def tab_correlation(df: pd.DataFrame) -> None:
    st.subheader("Pairwise Correlation Heatmap")

    corr_col1, corr_col2 = st.columns(2)
    with corr_col1:
        method = st.radio("Method", ["pearson", "spearman"], horizontal=True,
                          key="corr_method")
    with corr_col2:
        selected_cols = st.multiselect(
            "Metrics",
            _available(df, ANALYSIS_METRICS),
            default=_available(df, [
                "total_WD", "left_b", "right_b",
                "left_max_slope_deg", "right_max_slope_deg",
                "A_W", "A_S", "A_b", "skewness", "bottom_elev_m",
            ]),
            format_func=_label,
            key="corr_cols",
        )

    if len(selected_cols) < 2:
        st.info("Select at least 2 metrics.")
        return

    corr = df[selected_cols].corr(method=method)
    labels = [_label(c) for c in selected_cols]

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels, y=labels,
        zmin=-1, zmax=1,
        colorscale="RdBu",
        reversescale=True,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont_size=10,
        hoverongaps=False,
    ))
    fig.update_layout(
        title=f"{method.capitalize()} correlation matrix",
        height=max(400, 50 * len(selected_cols)),
        xaxis_tickangle=-35,
    )
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

    # Strongest correlations
    st.markdown("#### Top correlations (|r| > 0.4)")
    pairs = []
    for i, c1 in enumerate(selected_cols):
        for c2 in selected_cols[i + 1:]:
            r = corr.loc[c1, c2]
            if abs(r) > 0.4:
                pairs.append({"Metric A": _label(c1), "Metric B": _label(c2),
                               "r": round(r, 3)})
    if pairs:
        st.dataframe(
            pd.DataFrame(pairs).sort_values("r", key=abs, ascending=False),
            width="stretch", hide_index=True,
        )
    else:
        st.caption("No pairs with |r| > 0.4 found.")


# ── tab: side comparison ─────────────────────────────────────────────────────

def tab_sides(df: pd.DataFrame, n_bins: int) -> None:
    st.subheader("Left vs. Right Flank Comparison")
    dir_order, dir_colors = direction_scheme(n_bins)
    st.caption(
        "Each point is one cross-section. Points above the diagonal → "
        "left flank is larger; below → right flank is larger."
    )

    SIDE_PAIRS = [
        (f"left_{k}", f"right_{k}", lbl) for k, lbl in FLANK_METRIC_LABELS.items()
    ]
    available_pairs = [(l, r, lbl) for l, r, lbl in SIDE_PAIRS
                       if l in df.columns and r in df.columns]

    if not available_pairs:
        st.warning("No left/right metric pairs found.")
        return

    pair_labels = [lbl for _, _, lbl in available_pairs]
    chosen = st.selectbox("Metric", pair_labels, key="sides_metric")
    l_col, r_col, _ = available_pairs[pair_labels.index(chosen)]

    color_by_dir = st.checkbox("Colour by valley direction", True, key="sides_dir")
    df2 = df.dropna(subset=[l_col, r_col]).copy()
    if color_by_dir and "valley_bearing_deg" in df2.columns:
        df2["direction"] = classify_bearing(df2["valley_bearing_deg"], n_bins=n_bins)
        df2 = df2.dropna(subset=["direction"])
        color_col = "direction"
        cmap = dir_colors
    else:
        color_col = None
        cmap = None

    xy_min = min(df2[l_col].min(), df2[r_col].min())
    xy_max = max(df2[l_col].max(), df2[r_col].max())

    fig = px.scatter(
        df2, x=l_col, y=r_col,
        color=color_col, color_discrete_map=cmap,
        opacity=0.6,
        labels={l_col: f"Left {chosen}", r_col: f"Right {chosen}"},
        hover_data={c: True for c in ["valley_pct", "line_idx"] if c in df2.columns},
    )
    # Manual trendline
    _xv, _yv = df2[l_col].values, df2[r_col].values
    _ok = np.isfinite(_xv) & np.isfinite(_yv)
    if _ok.sum() > 2:
        _m, _b = np.polyfit(_xv[_ok], _yv[_ok], 1)
        _xs = np.linspace(_xv[_ok].min(), _xv[_ok].max(), 100)
        fig.add_trace(go.Scatter(
            x=_xs, y=_m * _xs + _b,
            mode="lines", line=dict(color="white", width=1.5, dash="dot"),
            name="trend", showlegend=False,
        ))
    fig.add_shape(type="line", x0=xy_min, x1=xy_max, y0=xy_min, y1=xy_max,
                  line=dict(color="white", dash="dash", width=1.5))
    fig.update_layout(height=480)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

    # Wilcoxon signed-rank test
    from scipy.stats import wilcoxon
    try:
        stat, p = wilcoxon(df2[l_col], df2[r_col], nan_policy="omit")
        mean_diff = (df2[l_col] - df2[r_col]).mean()
        st.caption(
            f"Wilcoxon signed-rank test: W = {stat:.1f}, p = {p:.4f}  |  "
            f"Mean difference (L − R) = {mean_diff:.3f}"
        )
    except Exception:
        pass


# ── tab: flank direction ─────────────────────────────────────────────────────

def tab_flank_direction(df_flanks: pd.DataFrame, n_bins: int) -> None:
    st.subheader("Flank Direction Analysis")
    dir_order, dir_colors = direction_scheme(n_bins)
    st.caption(
        "Each cross-section contributes **two** data points — one per flank — "
        "each classified by the bearing from the shoulder down to the valley bottom "
        f"(i.e. the downslope direction / aspect), into {n_bins} sectors labelled by their "
        "centre direction. This lets you compare, e.g., all north-facing flanks "
        "regardless of which valley or which side of the section they came from."
    )

    avail_metrics = [c for c in FLANK_METRICS if c in df_flanks.columns]
    if not avail_metrics:
        st.warning("No flank metrics found in the data.")
        return

    # ── controls ──────────────────────────────────────────────────────────
    fd_c1, fd_c2, fd_c3 = st.columns(3)
    with fd_c1:
        metric = st.selectbox(
            "Metric",
            avail_metrics,
            format_func=lambda c: FLANK_METRIC_LABELS.get(c, c),
            key="fd_metric",
        )
    with fd_c2:
        chart_type = st.radio("Chart", ["KDE", "Violin", "Box"],
                              horizontal=True, key="fd_chart")
    # Auto-default R² to 0.9 for b-related metrics (shape exponent depends on fit quality)
    _B_METRICS = {"b", "left_b", "right_b", "A_b"}
    _r2_auto = 0.9 if metric in _B_METRICS else 0.0
    if st.session_state.get("_fd_r2_prev_metric") != metric:
        st.session_state["fd_r2"] = _r2_auto
        st.session_state["_fd_r2_prev_metric"] = metric
    with fd_c3:
        min_r2 = st.slider("Min R² filter", 0.0, 1.0, _r2_auto, 0.05, key="fd_r2")

    df2 = df_flanks.copy()
    df2["flank_direction"] = classify_bearing(df2["flank_bearing_deg"], n_bins=n_bins)
    if min_r2 > 0 and "r2" in df2.columns:
        df2 = df2[df2["r2"].fillna(-1) >= min_r2]

    df2 = df2.dropna(subset=["flank_direction", metric])
    df2 = df2[df2["flank_direction"].isin(dir_order)]

    if df2.empty:
        st.warning("No data after filtering.")
        return

    label_m = FLANK_METRIC_LABELS.get(metric, metric)

    # ── main chart ────────────────────────────────────────────────────────
    if chart_type == "KDE":
        from scipy.stats import gaussian_kde as _gkde
        fig = go.Figure()
        _pool = np.concatenate([df2[df2["flank_direction"] == d][metric].dropna().values
                                 for d in dir_order
                                 if len(df2[df2["flank_direction"] == d][metric].dropna()) >= 3
                                 ] or [np.array([])])
        _x_hi = float(np.percentile(_pool, 99)) if len(_pool) else None
        for d in dir_order:
            vals = df2[df2["flank_direction"] == d][metric].dropna().values
            if len(vals) < 3:
                continue
            n = len(vals)
            kde = _gkde(vals)
            x_end = min(float(vals.max()), _x_hi) if _x_hi is not None else float(vals.max())
            x_rng = np.linspace(float(vals.min()), x_end, 300)
            y_rng = kde(x_rng)
            c_hex = dir_colors[d].lstrip("#")
            r, g, b_ = int(c_hex[0:2], 16), int(c_hex[2:4], 16), int(c_hex[4:6], 16)
            fig.add_trace(go.Scatter(
                x=x_rng, y=y_rng,
                mode="lines",
                name=f"{d}  (n={n})",
                line=dict(color=dir_colors[d], width=2.5),
                fill="tozeroy",
                fillcolor=f"rgba({r},{g},{b_},0.18)",
            ))
        fig.update_layout(
            xaxis_title=label_m,
            yaxis_title="Density",
            title=f"{label_m} by flank facing direction",
            legend_title="Direction",
            height=430,
        )
        if _x_hi is not None:
            fig.update_xaxes(range=[float(_pool.min()), _x_hi])

    elif chart_type == "Violin":
        fig = go.Figure()
        for d in dir_order:
            sub = df2[df2["flank_direction"] == d][metric]
            if sub.empty:
                continue
            fig.add_trace(go.Violin(
                y=sub, name=f"{d}  (n={len(sub)})",
                box_visible=True, meanline_visible=True,
                fillcolor=dir_colors[d], line_color=dir_colors[d], opacity=0.72,
            ))
        fig.update_layout(yaxis_title=label_m,
                          title=f"{label_m} by flank facing direction", height=430)
    else:
        fig = go.Figure()
        for d in dir_order:
            sub = df2[df2["flank_direction"] == d][metric]
            if sub.empty:
                continue
            fig.add_trace(go.Box(
                y=sub, name=f"{d}  (n={len(sub)})",
                marker_color=dir_colors[d], boxmean="sd",
            ))
        fig.update_layout(yaxis_title=label_m,
                          title=f"{label_m} by flank facing direction", height=430)

    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

    # ── stats table ───────────────────────────────────────────────────────
    st.markdown("#### Per-direction statistics")
    stats_rows = []
    for d in dir_order:
        sub = df2[df2["flank_direction"] == d][metric]
        if sub.empty:
            continue
        stats_rows.append({
            "Direction": d, "n flanks": len(sub),
            "mean": round(sub.mean(), 3), "median": round(sub.median(), 3),
            "std": round(sub.std(), 3),
            "min": round(sub.min(), 3), "max": round(sub.max(), 3),
        })
    st.dataframe(pd.DataFrame(stats_rows), width="stretch", hide_index=True)

    # ── polar rose ────────────────────────────────────────────────────────
    st.markdown("#### Median value per 10° bearing bin (polar)")
    df2["bearing_bin"] = (df2["flank_bearing_deg"] // 10 * 10).astype(int)
    polar_df = df2.groupby("bearing_bin")[metric].median().reset_index()
    fig_p = go.Figure(go.Barpolar(
        r=polar_df[metric] - polar_df[metric].min() + 1e-9,
        theta=polar_df["bearing_bin"],
        width=10,
        marker_color=polar_df[metric],
        marker_colorscale="Viridis",
        hovertemplate=f"Bearing: %{{theta}}°<br>Median {label_m}: %{{customdata:.3f}}",
        customdata=polar_df[metric],
    ))
    fig_p.update_layout(
        polar=dict(angularaxis=dict(direction="clockwise", rotation=90)),
        title=f"Median {label_m} – 10° bearing bins",
        height=420,
    )
    st.plotly_chart(fig_p, width="stretch", config={"displaylogo": False})

    # ── map scatter: flanks coloured by direction ─────────────────────────
    if {"flank_lon", "flank_lat"}.issubset(df2.columns):
        st.markdown("#### Flank locations coloured by direction")
        df_map = df2.dropna(subset=["flank_lon", "flank_lat"])
        fig_m = px.scatter(
            df_map, x="flank_lon", y="flank_lat",
            color="flank_direction", color_discrete_map=dir_colors,
            category_orders={"flank_direction": dir_order},
            size_max=8, opacity=0.6,
            labels={"flank_lon": "Longitude", "flank_lat": "Latitude",
                    "flank_direction": "Direction"},
            hover_data={c: True for c in [metric, "valley_pct", "line_idx"]
                        if c in df_map.columns},
        )
        fig_m.update_layout(height=420, yaxis_scaleanchor="x")
        st.plotly_chart(fig_m, width="stretch", config={"displaylogo": False})


# ── main ─────────────────────────────────────────────────────────────────────

def tab_guide() -> None:
    st.subheader("Metrics Guide")
    st.caption("What every column means and how it is computed.")

    with st.expander("1. How a cross-section is built", expanded=True):
        st.markdown(
            """
- A valley line is digitised on the hillshade and smoothed into a spline. Cross-sections are cut
  **perpendicular** to the line, spread evenly along it (more sections on longer lines), and
  sampled from the LiDAR DEM over ± *half-length* (set at export).
- **Left / Right** are seen when looking along the digitised direction of the line.
  Left is the negative-distance end of the section.
- **Shoulders** are found separately on each side, scanning outward from the centre on a
  smoothed (Savitzky–Golay) profile. The search stops at the first ridge (local maximum).
  The shoulder is the first point past the steepest slope where the slope drops below
  *peak slope / c*.
- **Valley floor** is the minimum of a 4th-order polynomial fitted between the two shoulders.
  It gives a sub-pixel position and is robust to single noisy pixels. If the fit fails it falls
  back to the lowest sample.
- Each flank is then re-expressed as **X** (horizontal distance from the floor) and **Y**
  (height above the floor). Sections where no shoulder is found have empty metrics.
- Distances and widths are in metres; slopes are in degrees from horizontal.
"""
        )

    with st.expander("2. Per-flank metrics (columns `left_<name>` / `right_<name>`)", expanded=True):
        st.markdown(
            """
| Name | Meaning |
|---|---|
| `W` | **Width**: horizontal distance from the valley floor to the shoulder. |
| `D` | **Depth**: height of the shoulder-side flank above the floor (max Y on that flank). |
| `WD100` | **W/D** at 100 % of depth: `W / D`. Large = wide and gentle, small = narrow and steep. |
| `floor_width_m` | Distance from the floor to where the flank first reaches **10 % of D** (a proxy for how flat the floor is). |
| `WD10` | `floor_width_m / D`: the same W/D idea measured at the 10 % level. |
| `b` | **Shape exponent** of the fit `Y = a·Xᵇ` (bounded 0.1–6). `b≈1` is a straight V, `b≈2` is parabolic, `b>2` has a flat floor with steep walls (box or U shaped), and `b<1` is steep near the floor with a gentle shoulder. |
| `r2` | R² of the power-law fit. Low values mean `b` is unreliable. Filter on it, for example R² ≥ 0.9. |
| `mean_slope_deg` | Angle of the mean gradient between **20 % and 80 %** of D (the mid-wall). |
| `max_slope_deg` | Angle of the steepest single segment in that 20–80 % band (noise-sensitive). |
| `slope_p90/70/50/30/10_deg` | Mean slope over a ±5 % band around that fraction of D (e.g. `p90` = 85–95 % of D). It shows how steepness changes from floor (p10) to shoulder (p90). |
| `dist_h50_m`, `dist_h90_m` | Horizontal distance from the floor at which the flank first reaches 50 % / 90 % of the **lower** flank's depth (`min(D_left, D_right)`), so both walls are compared at the same height above the floor. |
"""
        )

    with st.expander("3. Section-level and asymmetry metrics", expanded=True):
        st.markdown(
            """
| Column | Meaning |
|---|---|
| `total_WD` | `(W_left + W_right) / mean(D_left, D_right)`: overall valley width-to-depth ratio. |
| `A_W` | **Width asymmetry** `(W_L − W_R) / (W_L + W_R)`, in −1…+1. Positive means the left flank is wider. |
| `A_S` | **Slope asymmetry**: `max_slope_L − max_slope_R` (degrees). Positive means the left flank is steeper. |
| `A_b` | **Shape asymmetry**: `b_L − b_R`. |
| `skewness` | Sample skewness of the *height-above-floor values* of the whole profile (left and right combined). It ignores horizontal position, so it describes how the heights are distributed (a lot of low ground versus a lot of high ground), not left/right asymmetry. |
| `diff_<name>` | **Left − Right** for every per-flank metric above (e.g. `diff_dist_h50_m`, `diff_slope_p70_deg`). The sign tells you which side is larger. |
"""
        )

    with st.expander("4. Position, geometry and direction", expanded=True):
        st.markdown(
            """
| Column | Meaning |
|---|---|
| `line_idx`, `section_id`, `section_in_line` | Which valley line, global section number, and index within its line. |
| `valley_pct` | Position along the line, 0 % (start) to 100 % (end). |
| `line_length_m` | Length of the valley line the section belongs to. |
| `bottom_elev_m` | Elevation of the fitted valley floor. |
| `left_dist_m`, `right_dist_m` | Signed distance of each shoulder from the section centre (m). |
| `cs_bearing_deg` | Azimuth of the cross-section line from its left end to its right end (0 = N, 90 = E, clockwise). |
| `valley_bearing_deg` | `cs_bearing + 90°`. This points **against** the digitising direction of the line: a line drawn heading east has `valley_bearing` 270°. |
| `left_bearing_deg`, `right_bearing_deg` | **Flank aspect**: azimuth from the shoulder **down to the valley floor** (the downslope direction). A flank whose aspect is N faces north. |
| `*_lon`, `*_lat`, `*_easting`, `*_northing` | Coordinates of the floor (`bottom_*`), the shoulders (`left_*`, `right_*`) and the section centre (`center_*`). |
"""
        )

    with st.expander("5. Using the app", expanded=False):
        st.markdown(
            """
- **Direction bins** (sidebar): number of equal compass sectors used for every direction
  comparison. Bins are centred on 000° and every 360/n degrees, and each is labelled with its
  centre (e.g. `N (000°)`, `S (180°)`).
- **Flank Direction** uses one row per flank, grouped by that flank's aspect, so all
  north-facing walls can be compared regardless of which side of the section they were on.
- **Direction (sections)** groups whole sections by the chosen bearing column.
- **Min fit R²** (sidebar) drops sections whose power-law fit is poor on either flank.
  The Flank Direction tab applies it per flank and defaults to 0.9 for `b`.
"""
        )


def main() -> None:
    st.set_page_config(layout="wide", page_title="Valley Section Analysis")
    st.title("Valley Cross-Section Analysis")

    df_raw, df_flanks_raw = sidebar_load()
    if df_raw is None:
        st.info("Enter the path to a valley_sections.csv file in the sidebar.")
        return

    n_bins = st.sidebar.slider(
        "Direction bins", 2, 16, 4, key="n_dir_bins",
        help="Number of equal compass sectors used wherever directions are compared. "
             "Bins are centred on 000° (N), then every 360/n degrees.",
    )

    df = sidebar_filters(df_raw)
    n_total = len(df_raw)
    n_filtered = len(df)
    st.caption(f"Showing **{n_filtered:,}** of {n_total:,} sections after filters.")

    if df.empty:
        st.warning("All sections filtered out — relax the filters.")
        return

    # Build / filter the flanks dataframe
    if df_flanks_raw is not None:
        # Filter flanks to only sections that survived the section filter
        keep_ids = set(df["section_id"]) if "section_id" in df.columns else None
        df_flanks = df_flanks_raw[
            df_flanks_raw["section_id"].isin(keep_ids)
        ] if keep_ids is not None else df_flanks_raw
    else:
        df_flanks = _melt_flanks(df)

    tabs = st.tabs([
        "Overview",
        "Flank Direction",
        "Direction (sections)",
        "Along Valley",
        "Left vs. Right",
        "Scatter",
        "Correlation",
        "Metrics Guide",
    ])

    with tabs[0]:
        tab_overview(df)
    with tabs[1]:
        tab_flank_direction(df_flanks, n_bins)
    with tabs[2]:
        tab_direction(df, n_bins)
    with tabs[3]:
        tab_along_valley(df)
    with tabs[4]:
        tab_sides(df, n_bins)
    with tabs[5]:
        tab_scatter(df, n_bins)
    with tabs[6]:
        tab_correlation(df)
    with tabs[7]:
        tab_guide()


if __name__ == "__main__":
    main()
