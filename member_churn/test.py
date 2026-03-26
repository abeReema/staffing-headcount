"""
Member Attrition Model — Shared Frailty (Mixed-Effects Survival)
=================================================================
Models member churn with a per-client random effect (frailty term).
Information is shared across clients via shrinkage toward the global curve.
New clients automatically fall back to the global curve until data accumulates.

Dependencies:
    pip install lifelines pandas numpy matplotlib

Data requirements:
    A DataFrame with columns:
        - member_id       : unique member identifier
        - client_id       : which client this member belongs to
        - duration_months : time from join to leave (or to today if still active)
        - observed        : 1 = member left, 0 = still active (censored)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines import CoxPHFitter

# ─────────────────────────────────────────────────────────────
# 1. SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────────────────────
# Each client has a different underlying attrition rate, modelled
# as a multiplier on a shared baseline Weibull distribution.
# This simulates the real-world scenario where clients differ but
# share a common underlying survival pattern.

np.random.seed(42)

CLIENT_PROFILES = {
    # client_id : (n_members, frailty_multiplier)
    # frailty > 1 → higher hazard → faster churn
    # frailty < 1 → lower hazard  → stickier members
    "client_A": (200, 0.6),   # large, low churn
    "client_B": (150, 1.0),   # medium, average churn
    "client_C": (80,  1.6),   # medium, high churn
    "client_D": (40,  0.8),   # small, low churn
    "client_E": (15,  1.4),   # tiny, high churn (sparse data → heavy shrinkage)
}

NOW          = 36    # months of observation window
BASE_SHAPE   = 0.75  # Weibull shape — front-loaded attrition (shape < 1)
BASE_SCALE   = 18    # Weibull scale in months

records = []
for client_id, (n, frailty) in CLIENT_PROFILES.items():
    for _ in range(n):
        # Each member joins at a random point in the observation window
        join_month = np.random.uniform(0, NOW - 1)

        # True tenure drawn from Weibull, scaled by client frailty
        # Higher frailty → shorter expected tenure
        true_tenure = np.random.weibull(BASE_SHAPE) * BASE_SCALE / frailty

        # Member is observed to leave only if tenure fits in the window
        if join_month + true_tenure <= NOW:
            duration = true_tenure
            observed = 1
        else:
            duration = NOW - join_month   # censored at end of window
            observed = 0

        records.append({
            "client_id":       client_id,
            "duration_months": round(max(duration, 0.1), 2),
            "observed":        observed,
        })

df = pd.DataFrame(records)
df["member_id"] = range(len(df))

print("── Dataset Summary ──")
summary = (
    df.groupby("client_id")
    .agg(
        n_members   =("member_id",       "count"),
        n_churned   =("observed",        "sum"),
        pct_churned =("observed",        "mean"),
        median_tenure=("duration_months","median"),
    )
    .assign(pct_churned=lambda x: (x["pct_churned"] * 100).round(1))
)
print(summary.to_string())
print(f"\nTotal: {len(df)} members | {df['observed'].sum()} churned\n")


# ─────────────────────────────────────────────────────────────
# 2. LOG-RANK TEST
# Confirms whether client curves are statistically different.
# If p > 0.05, client may not be a meaningful grouping variable.
# ─────────────────────────────────────────────────────────────
results = multivariate_logrank_test(
    df["duration_months"],
    df["client_id"],
    df["observed"],
)
print("── Log-Rank Test (are client curves significantly different?) ──")
print(f"  Test statistic : {results.test_statistic:.3f}")
print(f"  p-value        : {results.p_value:.4f}")
print(f"  Conclusion     : {'Curves differ significantly ✓' if results.p_value < 0.05 else 'No significant difference'}\n")


# ─────────────────────────────────────────────────────────────
# 3. SHARED FRAILTY MODEL via Cox PH with cluster term
# ─────────────────────────────────────────────────────────────
# lifelines implements the shared frailty model via CoxPHFitter
# with a `cluster_col` argument. This fits a Cox model where:
#   - Client dummies capture the fixed per-client shift (frailty estimates)
#   - Standard errors are corrected for within-cluster correlation
#   - Information is shared via the shared baseline hazard h₀(t)
#
# The shrinkage toward the global curve comes naturally: clients
# with thin data have uncertain frailty estimates that sit close
# to zero (i.e. close to the global baseline).

# One-hot encode client_id (drop one as reference category)
df_model = pd.get_dummies(df, columns=["client_id"], drop_first=True, dtype=float)

cox = CoxPHFitter()
cox.fit(
    df_model,
    duration_col  ="duration_months",
    event_col     ="observed",
    cluster_col   ="member_id",    # robust SEs — treats each member as a cluster
    formula       ="client_id_client_B + client_id_client_C + client_id_client_D + client_id_client_E",
)

print("── Cox Frailty Model Summary ──")
cox.print_summary(decimals=3)


# ─────────────────────────────────────────────────────────────
# 4. SURVIVAL CURVES PER CLIENT
# Using the fitted Cox model to predict survival curves for each
# client profile, sharing the baseline hazard across all clients.
# ─────────────────────────────────────────────────────────────

# Create a representative "profile" row for each client
client_profiles_pred = pd.DataFrame({
    "duration_months":         [1] * 5,
    "observed":                [1] * 5,
    "client_id_client_B":      [0, 1, 0, 0, 0],
    "client_id_client_C":      [0, 0, 1, 0, 0],
    "client_id_client_D":      [0, 0, 0, 1, 0],
    "client_id_client_E":      [0, 0, 0, 0, 1],
}, index=["client_A", "client_B", "client_C", "client_D", "client_E"])

survival_curves = cox.predict_survival_function(client_profiles_pred)
# Columns are now client labels; index is time


# ─────────────────────────────────────────────────────────────
# 5. INDIVIDUAL RISK SCORES
# P(member churns in next N months | survived to current tenure)
# ─────────────────────────────────────────────────────────────

def survival_at(curve_series, t):
    """Interpolate survival probability at time t from a survival curve Series."""
    idx = curve_series.index
    if t <= idx.min():
        return 1.0
    if t >= idx.max():
        return float(curve_series.iloc[-1])
    # Linear interpolation between surrounding time points
    lower = idx[idx <= t].max()
    upper = idx[idx >= t].min()
    s_lo  = curve_series[lower]
    s_hi  = curve_series[upper]
    if upper == lower:
        return float(s_lo)
    frac = (t - lower) / (upper - lower)
    return float(s_lo + frac * (s_hi - s_lo))


def churn_risk(client_id, current_tenure, horizon_months=3):
    """
    P(member churns in next `horizon_months` | survived to `current_tenure`).
    Falls back to client_A (reference/global) curve for unknown clients.
    """
    known_clients = survival_curves.columns.tolist()
    col = client_id if client_id in known_clients else "client_A"
    curve = survival_curves[col]

    s_now  = survival_at(curve, current_tenure)
    s_next = survival_at(curve, current_tenure + horizon_months)

    if s_now == 0:
        return 1.0
    return round(1 - s_next / s_now, 4)


# Apply risk scoring to all active members
active = df[df["observed"] == 0].copy()
active["risk_3mo"] = active.apply(
    lambda r: churn_risk(r["client_id"], r["duration_months"], horizon_months=3),
    axis=1,
)
active["risk_tier"] = pd.cut(
    active["risk_3mo"],
    bins=[0, 0.15, 0.30, 1.0],
    labels=["Low", "Medium", "High"],
)

print("\n── Active Member Risk Distribution by Client ──")
print(
    active.groupby(["client_id", "risk_tier"], observed=True)
    .size()
    .unstack(fill_value=0)
    .to_string()
)

print("\n── Highest Risk Active Members (top 10) ──")
print(
    active.nlargest(10, "risk_3mo")[
        ["member_id", "client_id", "duration_months", "risk_3mo", "risk_tier"]
    ].to_string(index=False)
)


# ─────────────────────────────────────────────────────────────
# 6. NEW CLIENT FALLBACK
# A brand new client has no observed data. We use client_A
# (reference category = global baseline) until enough data
# accumulates. In practice, set a threshold like 30+ churned
# members before fitting a client-specific frailty term.
# ─────────────────────────────────────────────────────────────

NEW_CLIENT_THRESHOLD = 30   # minimum churned members before using client curve

def get_curve_for_client(client_id, client_churn_count):
    """
    Returns the appropriate survival curve for a client.
    Uses global baseline if client is new or has thin data.
    """
    if client_id not in survival_curves.columns or client_churn_count < NEW_CLIENT_THRESHOLD:
        print(f"  [{client_id}] Using global baseline (only {client_churn_count} churned members)")
        return survival_curves["client_A"]   # reference = global baseline
    return survival_curves[client_id]

print("\n── New Client Fallback Demo ──")
for cid, count in [("client_new", 0), ("client_E", 5), ("client_B", 95)]:
    curve = get_curve_for_client(cid, count)


# ─────────────────────────────────────────────────────────────
# 7. HEADCOUNT FORECAST PER CLIENT
# ─────────────────────────────────────────────────────────────

def forecast_client(client_id, monthly_injections, n_months=24):
    """
    Forecast active members for a given client over n_months,
    given a constant monthly injection rate.
    """
    curve = survival_curves.get(client_id, survival_curves["client_A"])
    max_t = curve.index.max()

    forecasts = []
    for future_month in range(1, n_months + 1):
        total = 0
        for months_ago in range(1, future_month + 1):
            tenure = future_month - months_ago + 1
            s = survival_at(curve, min(tenure, max_t))
            total += monthly_injections * s
        forecasts.append({"month": future_month, "active_members": round(total)})
    return pd.DataFrame(forecasts)


# Example: each client receives 20 new members/month
forecast_all = pd.concat(
    {cid: forecast_client(cid, monthly_injections=20, n_months=24)
     for cid in ["client_A", "client_B", "client_C", "client_D", "client_E"]},
    names=["client_id"]
).reset_index(level=0).reset_index(drop=True)

print("\n── 24-Month Forecast at Month 12 and 24 (20 new members/month per client) ──")
print(
    forecast_all[forecast_all["month"].isin([12, 24])]
    .pivot(index="client_id", columns="month", values="active_members")
    .rename(columns={12: "month_12", 24: "month_24"})
    .to_string()
)


# ─────────────────────────────────────────────────────────────
# 8. CHARTS
# ─────────────────────────────────────────────────────────────

DARK_BG   = "#0f1117"
PANEL_BG  = "#1a1d27"
WHITE     = "#e8eaf0"
GREY      = "#8b8fa8"
COLORS    = ["#4f8ef7", "#3dd68c", "#f5a623", "#e05c5c", "#b57bee"]
CLIENT_LABELS = ["client_A", "client_B", "client_C", "client_D", "client_E"]

def style_ax(ax, title):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2e3147")
    ax.tick_params(colors=GREY, labelsize=9)
    ax.xaxis.label.set_color(GREY)
    ax.yaxis.label.set_color(GREY)
    ax.set_title(title, color=WHITE, fontsize=11, fontweight="bold", pad=10)
    ax.grid(color="#2e3147", linewidth=0.6, linestyle="--")

fig = plt.figure(figsize=(16, 14), facecolor=DARK_BG)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

# ── Panel 1: Survival curves per client (Cox model predictions) ──
ax1 = fig.add_subplot(gs[0, :])
style_ax(ax1, "Survival Curves by Client  (shared frailty Cox model)")

for i, cid in enumerate(CLIENT_LABELS):
    curve = survival_curves[cid]
    n     = CLIENT_PROFILES[cid][0]
    ax1.plot(curve.index, curve.values,
             color=COLORS[i], linewidth=2.2,
             label=f"{cid}  (n={n})")

ax1.set_xlabel("Tenure (months)")
ax1.set_ylabel("P(still active)")
ax1.set_ylim(0, 1.05)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax1.legend(facecolor=PANEL_BG, edgecolor="#2e3147", labelcolor=WHITE, fontsize=9)

# ── Panel 2: KM curves per client (raw, unstratified) for comparison ──
ax2 = fig.add_subplot(gs[1, 0])
style_ax(ax2, "Raw KM Curves by Client  (no shrinkage, for comparison)")

for i, cid in enumerate(CLIENT_LABELS):
    kmf = KaplanMeierFitter()
    sub = df[df["client_id"] == cid]
    kmf.fit(sub["duration_months"], sub["observed"], label=cid)
    kmf.plot_survival_function(ax=ax2, color=COLORS[i], linewidth=1.8, ci_show=False)

ax2.set_xlabel("Tenure (months)")
ax2.set_ylabel("P(still active)")
ax2.set_ylim(0, 1.05)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax2.get_legend().remove()

# ── Panel 3: 24-month headcount forecast per client ──
ax3 = fig.add_subplot(gs[1, 1])
style_ax(ax3, "24-Month Headcount Forecast per Client  (20 new/month)")

for i, cid in enumerate(CLIENT_LABELS):
    fc = forecast_all[forecast_all["client_id"] == cid]
    ax3.plot(fc["month"], fc["active_members"],
             color=COLORS[i], linewidth=2, label=cid)

ax3.set_xlabel("Month from now")
ax3.set_ylabel("Active members")
ax3.legend(facecolor=PANEL_BG, edgecolor="#2e3147", labelcolor=WHITE, fontsize=9)

fig.suptitle("Member Attrition Model  ·  Shared Frailty (Mixed-Effects Survival)",
             color=WHITE, fontsize=14, fontweight="bold", y=0.98)

plt.savefig("frailty_model.png", dpi=160, bbox_inches="tight", facecolor=DARK_BG)
print("\nChart saved to frailty_model.png")