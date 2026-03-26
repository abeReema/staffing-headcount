import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Team View", page_icon="📈")

# ─────────────────────────────────────────────────────────────
# THEME / GLOBAL CSS
# ─────────────────────────────────────────────────────────────

css_path = 'styles.txt'


with open(css_path, 'r') as file:
    css = file.read() 

st.markdown(css, unsafe_allow_html=True)

np.random.seed(42)

TEAMS = ["Midwest Health", "Pacific Care", "Northeast Partners",
         "Southern Alliance", "Great Lakes Co", "UHC Missouri"]

GUIDE_NAMES = {
    team: [f"Guide {chr(65+i)}" for i in range(np.random.randint(4, 9))]
    for team in TEAMS
}

# ── Team-level metrics ──
def get_team_metrics(team: str) -> dict:
    rng = np.random.default_rng(abs(hash(team)) % (2**31))
    n_guides        = int(rng.integers(5, 20))
    ideal_guides    = int(rng.integers(4, 22))
    pct_mature      = round(float(rng.uniform(0.1, 0.8)), 2)
    return {
        "n_guides":     n_guides,
        "ideal_guides": ideal_guides,
        "pct_mature":   pct_mature,
    }

# ── Projected ideal team size (24 months) ──
def get_team_forecast(team: str) -> pd.DataFrame:
    rng    = np.random.default_rng(abs(hash(team)) % (2**31))
    base   = rng.integers(6, 18)
    trend  = rng.uniform(-0.1, 0.4)
    noise  = rng.normal(0, 0.4, 24)
    months = np.arange(1, 25)
    values = base + trend * months + noise
    values = np.clip(values, 1, None)
    return pd.DataFrame({"month": months, "ideal_guides": values.round(1)})

st.markdown('<div class="page-title">Team View</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Staffing overview by client team</div>', unsafe_allow_html=True)

selected_team = st.selectbox("Select team", TEAMS, key="team_select_p1")
metrics       = get_team_metrics(selected_team)
forecast_df   = get_team_forecast(selected_team)

n_guides     = metrics["n_guides"]
ideal_guides = metrics["ideal_guides"]
diff         = n_guides - ideal_guides
pct_mature   = metrics["pct_mature"]

# ── Metric cards ──
st.markdown('<div class="section-header">Staffing Snapshot</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="metric-card" style="--accent:#3d6fff;">
        <div class="metric-label">Current Guides</div>
        <div class="metric-value">{n_guides}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    if diff > 0:
        badge = f'<span class="metric-badge badge-over">▲ {diff} Overstaffed</span>'
    elif diff < 0:
        badge = f'<span class="metric-badge badge-under">▼ {abs(diff)} Understaffed</span>'
    else:
        badge = ""
    st.markdown(f"""
    <div class="metric-card" style="--accent:#a78bfa;">
        <div class="metric-label">Ideal Guides</div>
        <div class="metric-value">{ideal_guides}</div>
        {badge}
    </div>
    """, unsafe_allow_html=True)

with c3:
    accent = "#f5a623" if pct_mature >= 0.5 else "#3d6fff"
    st.markdown(f"""
    <div class="metric-card" style="--accent:{accent};">
        <div class="metric-label">Guides w/ &gt;20% Mature Members</div>
        <div class="metric-value">{pct_mature:.0%}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Forecast chart ──
st.markdown('<div class="section-header">Projected Ideal Team Size</div>', unsafe_allow_html=True)

fig = go.Figure()

# Shaded area
fig.add_trace(go.Scatter(
    x=forecast_df["month"],
    y=forecast_df["ideal_guides"],
    fill="tozeroy",
    fillcolor="rgba(61, 111, 255, 0.06)",
    line=dict(color="rgba(61,111,255,0)", width=0),
    showlegend=False,
    hoverinfo="skip",
))

# Main line
fig.add_trace(go.Scatter(
    x=forecast_df["month"],
    y=forecast_df["ideal_guides"],
    mode="lines+markers",
    line=dict(color="#3d6fff", width=2.5),
    marker=dict(size=5, color="#3d6fff",
                line=dict(color="#0c0e14", width=1.5)),
    name="Ideal guides",
    hovertemplate="Month %{x}<br>Ideal guides: %{y:.1f}<extra></extra>",
))

# Current guides reference line
fig.add_hline(
    y=n_guides,
    line_dash="dot",
    line_color="#555d72",
    line_width=1,
    annotation_text=f"Current: {n_guides}",
    annotation_font_color="#555d72",
    annotation_font_size=11,
)

fig.update_layout(
    paper_bgcolor="#0c0e14",
    plot_bgcolor="#0c0e14",
    font=dict(family="DM Mono, monospace", color="#c8cdd8", size=11),
    margin=dict(l=0, r=0, t=10, b=0),
    height=320,
    xaxis=dict(
        title="Month",
        tickfont=dict(size=10, color="#555d72"),
        gridcolor="#1e2130",
        zeroline=False,
        showline=False,
    ),
    yaxis=dict(
        title="Guides",
        tickfont=dict(size=10, color="#555d72"),
        gridcolor="#1e2130",
        zeroline=False,
        showline=False,
    ),
    showlegend=False,
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)
