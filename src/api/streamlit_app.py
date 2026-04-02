"""
Dashboard for AML Fraud Detection API

API must be running:
    uvicorn src.api.main:app --reload

PAGES:
Model Performance: deep dive on the best model
Model Comparison: all models side by side
Threshold Tuning: interactive slider to see how threshold affects metrics
Feature Importance: top features driving the model's decisions
Live Scoring: input transaction data and get a real-time prediction
"""

import math
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Page configuration

st.set_page_config(
    page_title="AML Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"



# ── Custom CSS ─────────────────────────────────────────────────────
# Streamlit has limited built-in styling. st.markdown with
# unsafe_allow_html=True lets you inject custom CSS.
# This gives the dashboard a dark industrial look.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #0d0f14; color: #e8eaf0; }
[data-testid="stSidebar"] { background-color: #13151c; border-right: 1px solid #1f2330; }

.metric-card {
    background: linear-gradient(135deg, #1a1d27 0%, #13151c 100%);
    border: 1px solid #252836; border-radius: 12px;
    padding: 1.2rem 1.4rem; position: relative; overflow: hidden;
}
.metric-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #00d4ff, #7b5ea7);
}
.metric-label {
    font-family: 'Space Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.12em; text-transform: uppercase; color: #6b7280; margin-bottom: 0.4rem;
}
.metric-value { font-family: 'Space Mono', monospace; font-size: 1.8rem; font-weight: 700; color: #e8eaf0; line-height: 1; }
.metric-sub   { font-size: 0.75rem; color: #4b5563; margin-top: 0.3rem; }

.risk-low    { color: #10b981; background: #10b98120; border: 1px solid #10b981; padding: 2px 10px; border-radius: 20px; font-family: 'Space Mono', monospace; font-size: 0.8rem; }
.risk-medium { color: #f59e0b; background: #f59e0b20; border: 1px solid #f59e0b; padding: 2px 10px; border-radius: 20px; font-family: 'Space Mono', monospace; font-size: 0.8rem; }
.risk-high   { color: #ef4444; background: #ef444420; border: 1px solid #ef4444; padding: 2px 10px; border-radius: 20px; font-family: 'Space Mono', monospace; font-size: 0.8rem; }

.page-header { font-family: 'Space Mono', monospace; font-size: 1.05rem; letter-spacing: 0.2em; text-transform: uppercase; color: #00d4ff; margin-bottom: 0.15rem; }
.page-title  { font-size: 2.2rem; font-weight: 600; color: #e8eaf0; margin-bottom: 0.5rem; line-height: 1.1; }
.page-sub    { color: #6b7280; font-size: 0.95rem; }
.divider     { border: none; border-top: 1px solid #1f2330; margin: 1.5rem 0; }

.stButton > button {
    background: linear-gradient(135deg, #00d4ff20, #7b5ea720);
    border: 1px solid #00d4ff60; color: #00d4ff;
    font-family: 'Space Mono', monospace; font-size: 0.75rem;
    letter-spacing: 0.1em; text-transform: uppercase; border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────

# Shared Plotly layout — dark theme applied to every chart
LAYOUT = dict(
    paper_bgcolor="#0d0f14",
    plot_bgcolor="#13151c",
    font=dict(family="DM Sans", color="#9ca3af", size=11),
    xaxis=dict(gridcolor="#1f2330", linecolor="#1f2330", zerolinecolor="#1f2330"),
    yaxis=dict(gridcolor="#1f2330", linecolor="#1f2330", zerolinecolor="#1f2330"),
    margin=dict(l=10, r=10, t=30, b=10),
)

# Each model gets a consistent colour and display name throughout the app
MODEL_META = {
    "logistic_regression": {"label": "Logistic Regression", "color": "#f59e0b"},
    "random_forest":       {"label": "Random Forest",       "color": "#7b5ea7"},
    "lightgbm":            {"label": "LightGBM",            "color": "#00d4ff"},
}

def model_label(name): return MODEL_META.get(name, {}).get("label", name)
def model_color(name): return MODEL_META.get(name, {}).get("color", "#9ca3af")


def api_get(endpoint: str):
    """
    Call a GET endpoint on the FastAPI backend.
    Returns the parsed JSON dict, or None if the server is unreachable.
    Streamlit will show an error message on non-connection failures.
    """
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return None  # Server not running — handled per-page
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(endpoint: str, payload: dict, params: dict = None):
    """Call a POST endpoint. payload is serialised to JSON automatically."""
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=payload, params=params, timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def metric_card(label: str, value: str, sub: str = ""):
    """
    Render a styled KPI card using HTML.
    Streamlit doesn't have a great built-in metric card,
    so we inject a small HTML block via st.markdown.
    """
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
    </div>
    """, unsafe_allow_html=True)

def hex_to_rgba(hex_color: str, alpha: float = 0.06) -> str:
    """Convert a 6-digit hex color to rgba() string for Plotly fillcolor."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ── Sidebar ────────────────────────────────────────────────────────
# st.sidebar.* renders content in the left panel.
# Everything inside the `with st.sidebar:` block goes there.

with st.sidebar:
    st.markdown('<div class="page-header">AML System</div>', unsafe_allow_html=True)
    st.markdown("### Fraud Detection")
    st.markdown("---")

    # Check API health and show status indicator
    health = api_get("/health")
    if health:
        color = "#10b981" if health["model_loaded"] else "#ef4444"
        label = "● Online" if health["model_loaded"] else "● Degraded"
        st.markdown(f'<span style="color:{color}; font-family:Space Mono; font-size:0.8rem;">{label}</span>', unsafe_allow_html=True)
        st.caption(f"Serving: `{health['model_version']}`")
    else:
        st.markdown('<span style="color:#ef4444; font-family:Space Mono; font-size:0.8rem;">● API Offline</span>', unsafe_allow_html=True)
        st.caption("Run: uvicorn src.api.main:app --reload")

    st.markdown("---")
    # st.radio creates a group of radio buttons.
    # The selected value is returned and stored in `page`.
    # The entire script reruns when the user picks a different page,
    # so the if/elif block below automatically shows the right content.
    page = st.radio(
        "Navigation",
        ["Model Performance", "Model Comparison", "Threshold Tuning",
         "Feature Importance", "Live Scoring"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:#4b5563; font-family:Space Mono; line-height:1.8;">
    MODELS<br>Logistic Regression<br>Random Forest<br>LightGBM<br><br>
    STACK<br>FastAPI · Streamlit<br>Plotly · scikit-learn
    </div>
    """, unsafe_allow_html=True)

# ── Page header (shown on every page) ─────────────────────────────
st.markdown('<div class="page-header">Anti-Money Laundering</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Fraud Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Multi-model comparison · LR · Random Forest · LightGBM · Real-time scoring</div>', unsafe_allow_html=True)
st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: Model Performance
# Shows the best model's metrics in detail — ROC, PR curve,
# confusion matrix, and the five key numbers.
# ══════════════════════════════════════════════════════════════════
if page == "Model Performance":
    metrics = api_get("/metrics")
    if not metrics:
        st.warning("API is offline. Start it with: `uvicorn src.api.main:app --reload`")
        st.stop()  # st.stop() halts rendering the rest of the page

    name = metrics.get("model_name", "best model")
    st.markdown(f"### Best Model — `{model_label(name)}`")

    # st.columns(5) creates 5 equal-width columns side by side.
    # The `with c1:` block renders content in that column.
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("ROC-AUC",   f"{metrics['roc_auc']:.3f}",   "Area under ROC curve")
    with c2: metric_card("PR-AUC",    f"{metrics['pr_auc']:.3f}",    "Area under PR curve")
    with c3: metric_card("F1 Score",  f"{metrics['f1']:.3f}",        "Harmonic mean P/R")
    with c4: metric_card("Precision", f"{metrics['precision']:.3f}", "True fraud rate")
    with c5: metric_card("Recall",    f"{metrics['recall']:.3f}",    "Fraud caught rate")

    st.markdown("<br>", unsafe_allow_html=True)
    color = model_color(name)

    # Two charts side by side
    left, right = st.columns(2)

    with left:
        st.markdown("##### ROC Curve")
        # go.Figure() creates a Plotly figure.
        # go.Scatter adds a line. fill="tozeroy" shades the area under it.
        roc = metrics["roc_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(  # diagonal reference line (random classifier)
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#374151", dash="dash", width=1), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=roc["fpr"], y=roc["tpr"], mode="lines",
            line=dict(color=color, width=2.5),
            fill="tozeroy", fillcolor=f"rgba(0,212,255,0.06)",
            name=f"AUC = {metrics['roc_auc']:.3f}",
        ))
        fig.update_layout(**LAYOUT, xaxis_title="FPR", yaxis_title="TPR",
                          legend=dict(x=0.6, y=0.1, bgcolor="rgba(0,0,0,0)"), height=320)
        # use_container_width=True makes the chart fill its column
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("##### Precision-Recall Curve")
        pr = metrics["pr_curve"]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=pr["recall"], y=pr["precision"], mode="lines",
            line=dict(color=color, width=2.5),
            fill="tozeroy", fillcolor=f"rgba(0,212,255,0.06)",
            name=f"AP = {metrics['pr_auc']:.3f}",
        ))
        fig2.update_layout(**LAYOUT, xaxis_title="Recall", yaxis_title="Precision",
                           legend=dict(x=0.6, y=0.9, bgcolor="rgba(0,0,0,0)"), height=320)
        st.plotly_chart(fig2, use_container_width=True)

    # Confusion matrix heatmap
    st.markdown("##### Confusion Matrix")
    cm = metrics["confusion_matrix"]
    labels = ["Legitimate", "Laundering"]
    fig3 = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        text=[[str(v) for v in row] for row in cm],
        texttemplate="%{text}", textfont=dict(size=18, family="Space Mono"),
        colorscale=[[0, "#13151c"], [1, hex_to_rgba(color, alpha=0.25)]],showscale=False, xgap=3, ygap=3,
    ))
    fig3.update_layout(**LAYOUT, xaxis_title="Predicted", yaxis_title="Actual", height=300)
    col, _ = st.columns([1, 2])
    with col:
        st.plotly_chart(fig3, use_container_width=True)



# ══════════════════════════════════════════════════════════════════
# PAGE: Model Comparison
# All three models side by side — grouped bar chart, overlaid
# ROC/PR curves, confusion matrices, and a summary table.
# ══════════════════════════════════════════════════════════════════
elif page == "Model Comparison":
    comparison = api_get("/comparison")
    if not comparison:
        st.warning("API offline.")
        st.stop()

    best = comparison["best_model"]
    models_data = comparison["models"]
    model_names = list(models_data.keys())

    st.markdown("### Model Comparison")
    st.markdown(
        f"All three models trained on the same stratified split. "
        f"Best model selected by **PR-AUC** (most informative metric "
        f"for imbalanced fraud data): **{model_label(best)}**"
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Per-model KPI cards
    cols = st.columns(3)
    for i, name in enumerate(model_names):
        m = models_data[name]
        c = model_color(name)
        badge = " ★ BEST" if name == best else ""
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card" style="border-color:{c}40;">
                <div class="metric-label" style="color:{c};">{model_label(name)}{badge}</div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.6rem; margin-top:0.8rem;">
                    <div><div class="metric-label">ROC-AUC</div><div class="metric-value" style="font-size:1.3rem;">{m['roc_auc']:.3f}</div></div>
                    <div><div class="metric-label">PR-AUC</div><div class="metric-value" style="font-size:1.3rem;">{m['pr_auc']:.3f}</div></div>
                    <div><div class="metric-label">F1</div><div class="metric-value" style="font-size:1.3rem;">{m['f1']:.3f}</div></div>
                    <div><div class="metric-label">Recall</div><div class="metric-value" style="font-size:1.3rem;">{m['recall']:.3f}</div></div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Grouped bar chart — all metrics, all models
    st.markdown("##### Metrics Comparison")
    metric_keys   = ["roc_auc", "pr_auc", "f1", "precision", "recall"]
    metric_labels = ["ROC-AUC", "PR-AUC", "F1", "Precision", "Recall"]
    fig = go.Figure()
    for name in model_names:
        m = models_data[name]
        fig.add_trace(go.Bar(
            name=model_label(name), x=metric_labels,
            y=[m[k] for k in metric_keys],
            marker_color=model_color(name), marker_line_width=0,
            text=[f"{m[k]:.3f}" for k in metric_keys],
            textposition="outside", textfont=dict(family="Space Mono", size=9),
        ))
    bar_layout = {**LAYOUT, "barmode": "group", "height": 400,
                  "yaxis": dict(range=[0, 1.15], gridcolor="#1f2330", linecolor="#1f2330"),
                  "legend": dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.08)}
    fig.update_layout(**bar_layout)
    st.plotly_chart(fig, use_container_width=True)

    # Overlaid ROC and PR curves — fetch full curve data per model
    st.markdown("##### ROC & Precision-Recall Curves — All Models")
    per_model = {}
    for name in model_names:
        data = api_get(f"/metrics/{name}")
        if data:
            per_model[name] = data

    if per_model:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                     line=dict(color="#374151", dash="dash", width=1), showlegend=False))
        fig_pr = go.Figure()
        for name, m in per_model.items():
            roc = m["roc_curve"]
            pr  = m["pr_curve"]
            fig_roc.add_trace(go.Scatter(
                x=roc["fpr"], y=roc["tpr"], mode="lines",
                line=dict(color=model_color(name), width=2.5),
                name=f"{model_label(name)} (AUC={m['roc_auc']:.3f})",
            ))
            fig_pr.add_trace(go.Scatter(
                x=pr["recall"], y=pr["precision"], mode="lines",
                line=dict(color=model_color(name), width=2.5),
                name=f"{model_label(name)} (AP={m['pr_auc']:.3f})",
            ))
        fig_roc.update_layout(**LAYOUT, xaxis_title="FPR", yaxis_title="TPR",
                              legend=dict(bgcolor="rgba(0,0,0,0)", x=0.5, y=0.1), height=350)
        fig_pr.update_layout(**LAYOUT, xaxis_title="Recall", yaxis_title="Precision",
                             legend=dict(bgcolor="rgba(0,0,0,0)", x=0.02, y=0.15), height=350)
        cl, cr = st.columns(2)
        with cl: st.plotly_chart(fig_roc, use_container_width=True)
        with cr: st.plotly_chart(fig_pr,  use_container_width=True)

    # Confusion matrices side by side
    st.markdown("##### Confusion Matrices")
    cm_cols = st.columns(3)
    for i, name in enumerate(model_names):
        m  = models_data[name]
        cm = m["confusion_matrix"]
        c  = model_color(name)
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=["Legit", "Laund."], y=["Legit", "Laund."],
            text=[[str(v) for v in row] for row in cm],
            texttemplate="%{text}", textfont=dict(size=16, family="Space Mono"),
            colorscale=[[0, "#13151c"], [1, hex_to_rgba(color, alpha=0.25)]], showscale=False, xgap=3, ygap=3,
        ))
        cm_layout = {**LAYOUT, "height": 260, "margin": dict(l=10, r=10, t=40, b=10),
                     "title": dict(text=model_label(name), font=dict(color=c, family="Space Mono", size=11)),
                     "xaxis_title": "Predicted", "yaxis_title": "Actual"}
        fig_cm.update_layout(**cm_layout)
        with cm_cols[i]:
            st.plotly_chart(fig_cm, use_container_width=True)

    # Expandable raw table
    with st.expander("Full metrics table"):
        rows = [{"Model": model_label(n), "ROC-AUC": m["roc_auc"], "PR-AUC": m["pr_auc"],
                 "F1": m["f1"], "Precision": m["precision"], "Recall": m["recall"],
                 "Best": "★" if n == best else ""}
                for n, m in models_data.items()]
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: Threshold Tuning
# Moving a slider instantly recalculates and redraws everything —
# that's Streamlit's rerun model at work.
# ══════════════════════════════════════════════════════════════════
elif page == "Threshold Tuning":
    metrics = api_get("/metrics")
    if not metrics:
        st.warning("API offline.")
        st.stop()

    name = metrics.get("model_name", "best model")
    st.markdown(f"### Threshold Tuning — `{model_label(name)}`")
    st.markdown(
        "Move the slider to explore the precision/recall tradeoff. "
        "In AML, **recall** matters most — you want to catch as much fraud "
        "as possible, even if it means more false alarms for investigators to clear."
    )

    # st.slider returns the current value — the script reruns with a new
    # value every time the user moves it.
    threshold = st.slider("Classification Threshold", 0.01, 0.99, 0.50, 0.01, format="%.2f")

    pr = metrics["pr_curve"]
    thresholds = pr["thresholds"]
    precisions = pr["precision"]
    recalls    = pr["recall"]

    # Find the index in the stored threshold array closest to the slider value
    idx         = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - threshold))
    t_precision = precisions[idx]
    t_recall    = recalls[idx]
    t_f1        = 2 * t_precision * t_recall / (t_precision + t_recall + 1e-9)

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Threshold",  f"{threshold:.2f}")
    with c2: metric_card("Precision",  f"{t_precision:.3f}", "Of flagged, % are real")
    with c3: metric_card("Recall",     f"{t_recall:.3f}",    "% of fraud caught")
    with c4: metric_card("F1 Score",   f"{t_f1:.3f}")

    st.markdown("<br>", unsafe_allow_html=True)

    f1_vals = [2*p*r/(p+r+1e-9) for p, r in zip(precisions[:-1], recalls[:-1])]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=precisions[:-1], name="Precision",
                             mode="lines", line=dict(color="#00d4ff", width=2)))
    fig.add_trace(go.Scatter(x=thresholds, y=recalls[:-1],    name="Recall",
                             mode="lines", line=dict(color="#7b5ea7", width=2)))
    fig.add_trace(go.Scatter(x=thresholds, y=f1_vals,         name="F1",
                             mode="lines", line=dict(color="#f59e0b", width=2, dash="dot")))
    # Vertical line showing the currently selected threshold
    fig.add_vline(x=threshold, line_width=1.5, line_dash="dash", line_color="#ef4444",
                  annotation_text=f"t={threshold:.2f}",
                  annotation_font=dict(color="#ef4444", family="Space Mono", size=11))
    fig.update_layout(**LAYOUT, xaxis_title="Threshold", yaxis_title="Score",
                      legend=dict(bgcolor="rgba(0,0,0,0)"), height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"At threshold **{threshold:.2f}**: for every 100 flagged transactions, "
        f"~{t_precision*100:.0f} are genuine laundering. "
        f"You catch ~{t_recall*100:.0f}% of all laundering."
    )


# ══════════════════════════════════════════════════════════════════
# PAGE: Feature Importance
# Horizontal bar chart — what did the model actually learn?
# ══════════════════════════════════════════════════════════════════
elif page == "Feature Importance":
    fi_data  = api_get("/features?top_n=20")
    metrics  = api_get("/metrics")
    if not fi_data:
        st.warning("API offline.")
        st.stop()

    name = metrics.get("model_name", "best model") if metrics else "best model"
    st.markdown(f"### Feature Importance — `{model_label(name)}`")
    st.markdown(
        "Split-based importance: how many times each feature was used "
        "to make a split across all trees. Higher = the model leans on it more."
    )

    features = fi_data["features"]
    df_fi    = pd.DataFrame(features).sort_values("importance", ascending=True)
    color    = model_color(name)

    fig = go.Figure(go.Bar(
        x=df_fi["importance"], y=df_fi["feature"], orientation="h",
        marker=dict(color=df_fi["importance"],
                    colorscale=[[0, "#1a1d27"], [0.5, "#7b5ea7"], [1, color]],
                    showscale=False),
        text=df_fi["importance"].apply(lambda v: f"{v:.4f}" if v < 1 else f"{v:.0f}"),
        textposition="outside",
        textfont=dict(family="Space Mono", size=10, color="#6b7280"),
    ))
    fi_layout = {**LAYOUT, "height": max(350, len(df_fi) * 28), "xaxis_title": "Importance",
                 "yaxis": dict(gridcolor="#1f2330", linecolor="#1f2330",
                               tickfont=dict(family="DM Sans", size=11))}
    fig.update_layout(**fi_layout)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw data"):
        st.dataframe(pd.DataFrame(features).sort_values("importance", ascending=False),
                     use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════
# PAGE: Live Scoring
# Submit a real transaction to the API and see the fraud score.
# Presets load sensible example values so the demo works out of
# the box — low/medium/high risk scenarios.
# ══════════════════════════════════════════════════════════════════
elif page == "Live Scoring":
    st.markdown("### Live Transaction Scoring")
    st.markdown(
        "Fill in transaction features below — or load a preset scenario — "
        "then click **Score Transaction** to call the API and see the result."
    )

    threshold = st.slider("Decision Threshold", 0.01, 0.99, 0.50, 0.01, format="%.2f")

    # Three preset scenarios for demo purposes
    PRESETS = {
        " Low Risk — Normal transfer": dict(
            amount_usd=250.0, hour_sin=0.0, hour_cos=1.0,
            day_of_week=2, dow_sin=0.782, dow_cos=0.623,
            is_weekend=0, is_business_hours=1, is_unusual_hour=0,
            is_cross_border=0, currency_mismatch=0,
            from_account_avg_amount=300.0, from_account_std_amount=80.0,
            from_account_min_amount=50.0, from_account_max_amount=800.0,
            from_account_total_transactions=120, from_account_total_volume=36000.0,
            from_unique_counterparties=5, from_account_cross_border_pct=0.05,
            from_account_unusual_hour_pct=0.03,
            amount_z_score=0.3, amount_percentile=0.55,
            is_round_amount=0, is_high_value=0,
            hours_since_last_txn=48.0, is_rapid_succession=0, txns_same_day=1,
            from_pagerank=0.0001, to_pagerank=0.0002, from_out_degree=3,
            to_in_degree=5, pagerank_ratio=2.0, suspicious_signal_count=0,
            transaction_hour=14,
        ),
        " Medium Risk — Cross-border elevated velocity": dict(
            amount_usd=4200.0, hour_sin=-0.5, hour_cos=-0.866,
            day_of_week=5, dow_sin=-0.975, dow_cos=0.223,
            is_weekend=1, is_business_hours=0, is_unusual_hour=0,
            is_cross_border=1, currency_mismatch=1,
            from_account_avg_amount=800.0, from_account_std_amount=400.0,
            from_account_min_amount=50.0, from_account_max_amount=5000.0,
            from_account_total_transactions=60, from_account_total_volume=48000.0,
            from_unique_counterparties=12, from_account_cross_border_pct=0.45,
            from_account_unusual_hour_pct=0.2,
            amount_z_score=1.8, amount_percentile=0.82,
            is_round_amount=0, is_high_value=1,
            hours_since_last_txn=3.0, is_rapid_succession=0, txns_same_day=3,
            from_pagerank=0.0012, to_pagerank=0.0045, from_out_degree=9,
            to_in_degree=22, pagerank_ratio=3.75, suspicious_signal_count=3,
            transaction_hour=22,
        ),
        " High Risk — Suspicious layering pattern": dict(
            amount_usd=9900.0, hour_sin=-0.866, hour_cos=-0.5,
            day_of_week=6, dow_sin=-0.782, dow_cos=0.623,
            is_weekend=1, is_business_hours=0, is_unusual_hour=1,
            is_cross_border=1, currency_mismatch=1,
            from_account_avg_amount=500.0, from_account_std_amount=200.0,
            from_account_min_amount=100.0, from_account_max_amount=2000.0,
            from_account_total_transactions=14, from_account_total_volume=7000.0,
            from_unique_counterparties=2, from_account_cross_border_pct=0.85,
            from_account_unusual_hour_pct=0.71,
            amount_z_score=4.5, amount_percentile=0.99,
            is_round_amount=1, is_high_value=1,
            hours_since_last_txn=0.2, is_rapid_succession=1, txns_same_day=9,
            from_pagerank=0.0089, to_pagerank=0.0310, from_out_degree=24,
            to_in_degree=67, pagerank_ratio=3.48, suspicious_signal_count=7,
            transaction_hour=3,
        ),
    }

    preset = st.selectbox("Load preset scenario", ["— custom —"] + list(PRESETS.keys()))
    d = PRESETS.get(preset, PRESETS[" Medium Risk — Cross-border elevated velocity"])

    # st.form groups inputs so the API is only called when the
    # submit button is pressed — not on every individual widget change.
    with st.form("score_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Amount & Flags**")
            amount_usd        = st.number_input("Amount (USD)",        value=float(d["amount_usd"]), min_value=0.01)
            is_cross_border   = st.selectbox("Cross-border",           [0, 1], index=d["is_cross_border"])
            currency_mismatch = st.selectbox("Currency mismatch",      [0, 1], index=d["currency_mismatch"])
            is_round_amount   = st.selectbox("Round amount",           [0, 1], index=d["is_round_amount"])
            is_high_value     = st.selectbox("High value",             [0, 1], index=d["is_high_value"])
            suspicious_signal_count = st.number_input("Suspicious signals", value=int(d["suspicious_signal_count"]), min_value=0)

        with c2:
            st.markdown("**Time & Context**")
            day_of_week        = st.slider("Day of week (0=Mon)",  0, 6,   d["day_of_week"])
            is_weekend         = st.selectbox("Is weekend",        [0, 1], index=d["is_weekend"])
            is_business_hours  = st.selectbox("Business hours",    [0, 1], index=d["is_business_hours"])
            is_unusual_hour    = st.selectbox("Unusual hour",      [0, 1], index=d["is_unusual_hour"])
            hours_since_last   = st.number_input("Hours since last txn", value=float(d["hours_since_last_txn"]), min_value=0.0)
            is_rapid           = st.selectbox("Rapid succession",  [0, 1], index=d["is_rapid_succession"])
            txns_same_day      = st.number_input("Txns today",     value=int(d["txns_same_day"]), min_value=0)

        with c3:
            st.markdown("**Account & Network**")
            from_avg    = st.number_input("Account avg amount",   value=float(d["from_account_avg_amount"]))
            from_total  = st.number_input("Account total vol",    value=float(d["from_account_total_volume"]))
            from_txns   = st.number_input("Account total txns",   value=int(d["from_account_total_transactions"]), min_value=0)
            from_cp     = st.number_input("Unique counterparties",value=int(d["from_unique_counterparties"]), min_value=0)
            amount_z    = st.number_input("Amount z-score",       value=float(d["amount_z_score"]))
            from_pr     = st.number_input("From PageRank",        value=float(d["from_pagerank"]), format="%.5f")
            to_pr       = st.number_input("To PageRank",          value=float(d["to_pagerank"]),   format="%.5f")
            to_in_deg   = st.number_input("To in-degree",         value=int(d["to_in_degree"]),    min_value=0)

        submitted = st.form_submit_button("▶ SCORE TRANSACTION", use_container_width=True)

    if submitted:
        # Build the payload using the submitted form values.
        # Some fields the user didn't edit we pull from the preset defaults.
        angle_h = math.pi * 2 * (14 / 24)  # default to 2pm for hour encoding
        payload = {
            "amount_usd": amount_usd,
            "hour_sin": d["hour_sin"], "hour_cos": d["hour_cos"],
            "day_of_week": day_of_week,
            "dow_sin": d["dow_sin"], "dow_cos": d["dow_cos"],
            "is_weekend": is_weekend, "is_business_hours": is_business_hours,
            "is_unusual_hour": is_unusual_hour,
            "is_cross_border": is_cross_border, "currency_mismatch": currency_mismatch,
            "from_account_avg_amount": from_avg,
            "from_account_std_amount": float(d["from_account_std_amount"]),
            "from_account_min_amount": float(d["from_account_min_amount"]),
            "from_account_max_amount": float(d["from_account_max_amount"]),
            "from_account_total_transactions": from_txns,
            "from_account_total_volume": from_total,
            "from_unique_counterparties": from_cp,
            "from_account_cross_border_pct": float(d["from_account_cross_border_pct"]),
            "from_account_unusual_hour_pct": float(d["from_account_unusual_hour_pct"]),
            "amount_z_score": amount_z,
            "amount_percentile": float(d["amount_percentile"]),
            "is_round_amount": is_round_amount, "is_high_value": is_high_value,
            "hours_since_last_txn": hours_since_last,
            "is_rapid_succession": is_rapid, "txns_same_day": txns_same_day,
            "from_pagerank": from_pr, "to_pagerank": to_pr,
            "from_out_degree": int(d["from_out_degree"]),
            "to_in_degree": to_in_deg,
            "pagerank_ratio": round(to_pr / (from_pr + 1e-10), 4),
            "suspicious_signal_count": suspicious_signal_count,
            "transaction_hour": int(d["transaction_hour"]),
        }
           
        result = api_post("/predict", payload, params={"threshold": threshold})

        if result:
            prob     = result["fraud_probability"]
            risk     = result["risk_level"]
            is_laund = result["is_laundering"]

            risk_colors = {"LOW": "#10b981", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}
            color = risk_colors.get(risk, "#6b7280")

            # Gauge chart — visually shows the score 0–100%
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                number={"suffix": "%", "font": {"family": "Space Mono", "size": 42, "color": color}},
                delta={"reference": threshold * 100, "suffix": "% threshold"},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#374151", "tickfont": {"color": "#6b7280"}},
                    "bar": {"color": color, "thickness": 0.25},
                    "bgcolor": "#13151c", "borderwidth": 0,
                    "steps": [
                        {"range": [0,  30], "color": "rgba(16,185,129,0.08)"},
                        {"range": [30, 60], "color": "rgba(245,158,11,0.08)"},
                        {"range": [60,100], "color": "rgba(239,68,68,0.08)"},
                    ],
                    "threshold": {"line": {"color": "rgba(255,255,255,0.31)", "width": 2},
                                  "thickness": 0.75, "value": threshold * 100},
                },
            ))
            fig.update_layout(paper_bgcolor="#0d0f14", font=dict(color="#9ca3af"),
                              height=280, margin=dict(l=20, r=20, t=20, b=10))

            g, v, m = st.columns([2, 1, 1])
            with g: st.plotly_chart(fig, use_container_width=True)
            with v:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown(f"### {'🔴 LAUNDERING' if is_laund else '🟢 LEGITIMATE'}")
                st.markdown(f'<span class="risk-{risk.lower()}">{risk} RISK</span>', unsafe_allow_html=True)
            with m:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.metric("Probability", f"{prob:.1%}")
                st.metric("Threshold",   f"{threshold:.2f}")
        else:
            st.error("Could not reach the API. Is `uvicorn src.api.main:app --reload` running?")