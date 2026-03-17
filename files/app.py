"""
app.py  –  Predictive Infrastructure Monitoring Dashboard
Run:  streamlit run app.py

Architecture
------------
- Loads the trained model + scored CSV once at startup.
- Auto-refreshes every REFRESH_MS milliseconds (via st_autorefresh).
- Maintains a rolling window of the last WINDOW rows in session_state.
- Draws live charts, anomaly score graph, timeline, and agent actions.
- BONUS: uses rolling trend to predict next anomaly window.
"""

import time
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

from monitor import stream_metrics, load_artifacts
from agent   import DevOpsAgent

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Infra Monitor",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── constants ─────────────────────────────────────────────────────────────────
WINDOW      = 120       # rows to display in rolling charts
REFRESH_MS  = 800       # dashboard auto-refresh interval in ms
PREDICT_WIN = 10        # lookahead rows for "predicted anomaly" banner
SCORE_ALERT = 0.05      # anomaly_score threshold for alert

# ── custom CSS (dark terminal aesthetic) ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background: #080c14;
    color: #c8d6e5;
}
.stApp { background: #080c14; }

/* metric cards */
[data-testid="metric-container"] {
    background: #0d1117;
    border: 1px solid #1f2937;
    border-radius: 8px;
    padding: 16px 20px;
}
[data-testid="stMetricValue"]  { color: #38bdf8; font-size: 1.6rem !important; }
[data-testid="stMetricLabel"]  { color: #64748b; font-size: .75rem !important; }
[data-testid="stMetricDelta"]  { font-size: .8rem !important; }

/* headers */
h1 { font-family: 'Syne', sans-serif; color: #f0f6ff; letter-spacing: -1px; }
h2, h3 { font-family: 'Syne', sans-serif; color: #94a3b8; }

/* alert boxes */
.alert-red  { background:#1f0a0a; border-left:4px solid #ef4444;
              padding:12px 16px; border-radius:6px; margin:6px 0;
              color:#fca5a5; font-size:.85rem; }
.alert-amber{ background:#1a1400; border-left:4px solid #f59e0b;
              padding:12px 16px; border-radius:6px; margin:6px 0;
              color:#fcd34d; font-size:.85rem; }
.alert-green{ background:#071a0d; border-left:4px solid #22c55e;
              padding:12px 16px; border-radius:6px; margin:6px 0;
              color:#86efac; font-size:.85rem; }
.badge-anomaly{ background:#dc2626; color:#fff; padding:2px 8px;
                border-radius:4px; font-size:.75rem; font-weight:700; }
.badge-normal { background:#166534; color:#dcfce7; padding:2px 8px;
                border-radius:4px; font-size:.75rem; }

/* sidebar */
[data-testid="stSidebar"]{ background:#0d1117 !important; }

/* dividers */
hr { border-color: #1f2937; }
</style>
""", unsafe_allow_html=True)


# ── load artifacts (cached) ───────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return load_artifacts()

@st.cache_data
def get_dataset():
    df = pd.read_csv("infrastructure_metrics.csv", parse_dates=["timestamp"])
    return df


# ── session-state bootstrap ───────────────────────────────────────────────────
def init_state():
    if "idx" not in st.session_state:
        st.session_state.idx       = 0
        st.session_state.history   = []   # list of row dicts
        st.session_state.alerts    = []   # recent alert strings
        st.session_state.gen       = None


# ── auto-refresh (every REFRESH_MS ms) ───────────────────────────────────────
st_autorefresh(interval=REFRESH_MS, key="dashboard_refresh")


def main():
    init_state()
    model, scaler = get_model()
    df = get_dataset()
    agent = DevOpsAgent()

    # ── initialise / advance the streaming generator ──────────────────────────
    if st.session_state.gen is None:
        st.session_state.gen = stream_metrics(df, model, scaler)

    try:
        row = next(st.session_state.gen)
        st.session_state.history.append(row)
        st.session_state.idx += 1

        if row["anomaly_label"] == 1:
            ts  = pd.Timestamp(row["timestamp"]).strftime("%H:%M:%S")
            msg = (f"[{ts}] ⚠ ANOMALY  score={row['anomaly_score']:.3f}  "
                   f"flags={row['flags']}")
            st.session_state.alerts.insert(0, msg)
            st.session_state.alerts = st.session_state.alerts[:15]
    except StopIteration:
        st.session_state.gen = None   # restart if we exhaust the dataset

    hist = st.session_state.history[-WINDOW:]
    if not hist:
        st.info("Waiting for data …")
        return

    hdf = pd.DataFrame(hist)

    # ── predict upcoming anomaly (BONUS) ─────────────────────────────────────
    future_rows = df.iloc[
        min(st.session_state.idx, len(df) - PREDICT_WIN) :
        min(st.session_state.idx + PREDICT_WIN, len(df))
    ]
    predicted_soon = False
    if not future_rows.empty and "predicted" in future_rows.columns:
        predicted_soon = (future_rows["predicted"] == 1).any()

    # ── header ────────────────────────────────────────────────────────────────
    col_title, col_badge = st.columns([6, 1])
    with col_title:
        st.markdown("# 🖥️  Predictive Infrastructure Monitor")
    with col_badge:
        status = "🟢 HEALTHY" if hdf["anomaly_label"].iloc[-1] == 0 else "🔴 ANOMALY"
        st.markdown(f"<br><span style='font-size:1.1rem'>{status}</span>",
                    unsafe_allow_html=True)

    if predicted_soon:
        st.markdown(
            '<div class="alert-amber">🔮 <b>PREDICTION:</b> '
            'Anomaly likely in the next ~2 minutes based on trend analysis.</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── KPI metrics row ───────────────────────────────────────────────────────
    last = hdf.iloc[-1]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("CPU Usage",        f"{last['cpu_usage']:.1f} %",
              f"{last['cpu_usage'] - hdf['cpu_usage'].iloc[-2]:.1f}" if len(hdf)>1 else "")
    m2.metric("RAM Usage",        f"{last['ram_usage']:.1f} %",
              f"{last['ram_usage'] - hdf['ram_usage'].iloc[-2]:.1f}" if len(hdf)>1 else "")
    m3.metric("Network Latency",  f"{last['network_latency']:.0f} ms")
    m4.metric("Request Rate",     f"{last['request_rate']:.0f} r/s")
    m5.metric("Anomaly Score",    f"{last['anomaly_score']:.3f}",
              "⚠ HIGH" if last["anomaly_score"] > SCORE_ALERT else "✓ normal")

    st.markdown("---")

    # ── layout: left charts | right panel ────────────────────────────────────
    left, right = st.columns([3, 1])

    with left:
        # ── combined CPU / RAM / Latency chart ───────────────────────────────
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=("CPU Usage (%)", "RAM Usage (%)", "Network Latency (ms)"),
        )
        ts_idx = list(range(len(hdf)))

        # shade anomaly regions
        def shade_anomalies(fig, hdf, row_n):
            in_block = False
            for i, lbl in enumerate(hdf["anomaly_label"]):
                if lbl == 1 and not in_block:
                    x0, in_block = i, True
                elif lbl == 0 and in_block:
                    fig.add_vrect(x0=x0, x1=i, fillcolor="rgba(239,68,68,0.12)",
                                  line_width=0, row=row_n, col=1)
                    in_block = False
            if in_block:
                fig.add_vrect(x0=x0, x1=len(hdf)-1,
                              fillcolor="rgba(239,68,68,0.12)",
                              line_width=0, row=row_n, col=1)

        for metric, row_n, color in [
            ("cpu_usage",       1, "#38bdf8"),
            ("ram_usage",       2, "#a78bfa"),
            ("network_latency", 3, "#fb923c"),
        ]:
            fig.add_trace(go.Scatter(
                x=ts_idx, y=hdf[metric],
                mode="lines", line=dict(color=color, width=1.8),
                name=metric, showlegend=False,
            ), row=row_n, col=1)
            shade_anomalies(fig, hdf, row_n)

        fig.update_layout(
            height=420, paper_bgcolor="#080c14", plot_bgcolor="#0d1117",
            font=dict(color="#94a3b8", size=11),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        fig.update_xaxes(showgrid=False, color="#374151")
        fig.update_yaxes(gridcolor="#1f2937", color="#374151")
        st.plotly_chart(fig, use_container_width=True)

        # ── anomaly score time series ─────────────────────────────────────────
        st.subheader("Anomaly Score")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=ts_idx, y=hdf["anomaly_score"],
            fill="tozeroy", fillcolor="rgba(239,68,68,0.08)",
            line=dict(color="#ef4444", width=1.5),
            name="Anomaly Score",
        ))
        fig2.add_hline(y=SCORE_ALERT, line_dash="dash",
                       line_color="#f59e0b", annotation_text="Alert Threshold")
        fig2.update_layout(
            height=180, paper_bgcolor="#080c14", plot_bgcolor="#0d1117",
            font=dict(color="#94a3b8", size=11),
            margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
        )
        fig2.update_xaxes(showgrid=False, color="#374151")
        fig2.update_yaxes(gridcolor="#1f2937", color="#374151")
        st.plotly_chart(fig2, use_container_width=True)

        # ── anomaly timeline ──────────────────────────────────────────────────
        st.subheader("Anomaly Timeline")
        timeline_df = hdf[hdf["anomaly_label"] == 1].copy()
        if not timeline_df.empty:
            fig3 = go.Figure(go.Scatter(
                x=timeline_df.index.tolist(),
                y=timeline_df["anomaly_score"],
                mode="markers",
                marker=dict(size=8, color="#ef4444", symbol="x"),
                text=[f"Score: {s:.3f}<br>Flags: {f}"
                      for s, f in zip(timeline_df["anomaly_score"],
                                       timeline_df["flags"])],
                hovertemplate="%{text}<extra></extra>",
            ))
            fig3.update_layout(
                height=150, paper_bgcolor="#080c14", plot_bgcolor="#0d1117",
                font=dict(color="#94a3b8", size=11),
                margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
            )
            fig3.update_xaxes(showgrid=False)
            fig3.update_yaxes(gridcolor="#1f2937")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.markdown('<div class="alert-green">No anomalies in current window.</div>',
                        unsafe_allow_html=True)

    with right:
        # ── live alert feed ───────────────────────────────────────────────────
        st.subheader("⚠ Alert Feed")
        if st.session_state.alerts:
            for a in st.session_state.alerts[:8]:
                st.markdown(f'<div class="alert-red">{a}</div>',
                            unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-green">All systems normal.</div>',
                        unsafe_allow_html=True)

        st.markdown("---")

        # ── agent recommendations ─────────────────────────────────────────────
        st.subheader("🤖 Agent Actions")
        if last["anomaly_label"] == 1 and last.get("actions"):
            for action in last["actions"]:
                st.markdown(f'<div class="alert-amber">{action}</div>',
                            unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-green">No remediation needed.</div>',
                        unsafe_allow_html=True)

        st.markdown("---")

        # ── stats ─────────────────────────────────────────────────────────────
        st.subheader("📊 Window Stats")
        n_total   = len(hdf)
        n_anom    = hdf["anomaly_label"].sum()
        st.markdown(f"""
        <small>
        Samples shown : **{n_total}**<br>
        Anomalies     : **{n_anom}** ({n_anom/max(n_total,1)*100:.1f} %)<br>
        Max CPU       : **{hdf['cpu_usage'].max():.1f} %**<br>
        Max RAM       : **{hdf['ram_usage'].max():.1f} %**<br>
        Max Latency   : **{hdf['network_latency'].max():.0f} ms**<br>
        Max Score     : **{hdf['anomaly_score'].max():.3f}**
        </small>
        """, unsafe_allow_html=True)


main()
