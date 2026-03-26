import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Guide View", page_icon="📈")

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

# ── Guide-level member table ──
def get_guide_members(team: str, guide: str) -> pd.DataFrame:
    rng      = np.random.default_rng(abs(hash(team + guide)) % (2**31))
    n        = int(rng.integers(8, 25))
    ids      = [f"MBR-{rng.integers(10000, 99999)}" for _ in range(n)]
    days     = rng.integers(1, 180, n).tolist()
    r1       = np.clip(rng.beta(1.5, 4, n), 0, 1).round(3).tolist()
    r3       = np.clip(np.array(r1) * rng.uniform(0.8, 1.3, n), 0, 1).round(3).tolist()
    r6       = np.clip(np.array(r3) * rng.uniform(0.8, 1.3, n), 0, 1).round(3).tolist()
    high     = [r >= 0.5 for r in r1]
    df = pd.DataFrame({
        "member_id":                   ids,
        "business_days_since_engagement": days,
        "1_month_churn_risk":          r1,
        "3_month_churn_risk":          r3,
        "6_month_churn_risk":          r6,
        "high_risk":                   high,
    })
    return df.sort_values("1_month_churn_risk", ascending=False).reset_index(drop=True)


st.markdown('<div class="page-title">Guide View</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Member-level churn risk by guide</div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    selected_team = st.selectbox("Select team", TEAMS, key="team_select_p2")
with col_b:
    guides         = GUIDE_NAMES[selected_team]
    selected_guide = st.selectbox("Select guide", guides, key="guide_select")

members = get_guide_members(selected_team, selected_guide)

n_members  = len(members)
n_highrisk = members["high_risk"].sum()

# ── Quick stats ──
st.markdown('<div class="section-header">Guide Summary</div>', unsafe_allow_html=True)
s1, s2, s3 = st.columns(3)

with s1:
    st.markdown(f"""
    <div class="metric-card" style="--accent:#3d6fff;">
        <div class="metric-label">Total Members</div>
        <div class="metric-value">{n_members}</div>
    </div>
    """, unsafe_allow_html=True)

with s2:
    st.markdown(f"""
    <div class="metric-card" style="--accent:#e05c5c;">
        <div class="metric-label">High Risk Members</div>
        <div class="metric-value">{n_highrisk}</div>
    </div>
    """, unsafe_allow_html=True)

with s3:
    avg_risk = members["1_month_churn_risk"].mean()
    st.markdown(f"""
    <div class="metric-card" style="--accent:#f5a623;">
        <div class="metric-label">Avg 1-Month Risk</div>
        <div class="metric-value">{avg_risk:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Member table ──
st.markdown('<div class="section-header">Member Risk Table — sorted by 1-month churn risk</div>',
            unsafe_allow_html=True)

rows_html = ""

for _, row in members.iterrows():
    tr_class = "high-risk" if row["high_risk"] else ""

    risk_pill = (
        '<span class="risk-pill risk-high">● HIGH</span>'
        if row["high_risk"]
        else '<span class="risk-pill risk-low">○ low</span>'
    )

    rows_html += f"""
    <thread>
        <tr class="{tr_class}">
            <td>{row['member_id']}</td>
            <td>{row['business_days_since_engagement']}</td>
            <td class="risk-col">{row['1_month_churn_risk']:.3f}</td>
            <td>{row['3_month_churn_risk']:.3f}</td>
            <td>{row['6_month_churn_risk']:.3f}</td>
            <td>{risk_pill}</td>
        </tr>
    </thread>
    """

table_html = f"""
    <table class="member-table">
        <thead>
            <tr>
                <th>Member ID</th>
                <th>Biz Days Since Engagement</th>
                <th>1-Month Risk</th>
                <th>3-Month Risk</th>
                <th>6-Month Risk</th>
                <th>High Risk</th>
            </tr>
        </thead>

        {rows_html}
    </table>
"""
st.markdown(table_html, unsafe_allow_html=True)