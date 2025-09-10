"""
NPV sensitivity analysis across Discount Rate (5–15%) and Revenue Growth (0–10%).

How it works
------------
- Assumes a 10-year forecast with revenue growing by `g` per year.
- Converts revenue to Free Cash Flow (FCF) using simple operating assumptions.
- Computes NPV for every (discount_rate, growth) pair and plots a heatmap.

Dependencies: numpy, matplotlib
Run: python npv_sensitivity.py (or run in a notebook cell)
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# User-adjustable assumptions
# -----------------------------
horizon_years = 10
initial_investment = 25_000_000  # positive = upfront cash outflow at t0

# Operating model
rev_year1 = 10_000_000          # revenue in Year 1
ebitda_margin = 0.25            # EBITDA as % of revenue
tax_rate = 0.20                 # tax on positive EBIT only
capex_pct_of_revenue = 0.08     # CAPEX % of revenue
actionable_note = "Adjust assumptions above to fit your case."

# Networking capital: percentage of revenue (stock + receivables - payables)
# We model cash impact as change in NWC each year.
nwc_pct_of_revenue = 0.05

# Sensitivity ranges (inclusive)
discount_rates = np.linspace(0.05, 0.15, 21)   # 5% to 15% in 0.5% steps
revenue_growths = np.linspace(0.00, 0.10, 21)  # 0% to 10% in 0.5% steps

# -----------------------------
# Vectorized cash-flow model
# -----------------------------
years = np.arange(1, horizon_years + 1)               # shape (T,)
G = revenue_growths.size
R = discount_rates.size
T = years.size

# Revenue matrix for each growth rate g across time t (shape: G x T)
# Revenue_t = rev_year1 * (1+g)^(t-1)
revenue = rev_year1 * (1.0 + revenue_growths[:, None]) ** (years[None, :] - 1)

# Previous year's revenue for ΔNWC calculation (shape: G x T)
# Define year-0 revenue so that Δ for Year 1 is consistent.
rev_year0 = rev_year1 / (1.0 + revenue_growths)  # shape (G,)
prev_revenue = np.concatenate([rev_year0[:, None], revenue[:, :-1]], axis=1)

# Operating line items
EBITDA = revenue * ebitda_margin
Depreciation = capex_pct_of_revenue * revenue
EBIT = EBITDA - Depreciation
Taxes = np.maximum(EBIT, 0.0) * tax_rate  # no tax shield on losses in this simple model
NOPAT = EBIT - Taxes
CAPEX = capex_pct_of_revenue * revenue
Delta_NWC = nwc_pct_of_revenue * (revenue - prev_revenue)

# Free Cash Flow (FCF) each year (shape: G x T)
FCF = NOPAT + Depreciation - CAPEX - Delta_NWC

# Discount factors for each rate across time (shape: R x T)
disc = (1.0 / (1.0 + discount_rates)[:, None]) ** years[None, :]

# Present value sum for every (r, g) pair -> (R x G)
PV_sum = (disc[:, None, :] * FCF[None, :, :]).sum(axis=2)

# NPV grid (R x G): subtract the initial investment at t0
NPV_grid = PV_sum - initial_investment

# -----------------------------
# Heatmap plot
# -----------------------------
fig, ax = plt.subplots(figsize=(9, 6))

# Use imshow with explicit axes in percentage units
extent = [revenue_growths[0] * 100, revenue_growths[-1] * 100, 
          discount_rates[0] * 100, discount_rates[-1] * 100]
im = ax.imshow(NPV_grid, origin='lower', aspect='auto', extent=extent)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('NPV')

ax.set_title('NPV Sensitivity Heatmap')
ax.set_xlabel('Revenue growth (%)')
ax.set_ylabel('Discount rate (%)')

# Optional: show zero-NPV contour
try:
    # Build coordinates for contour in the same units
    from matplotlib import ticker
    X, Y = np.meshgrid(revenue_growths * 100, discount_rates * 100)
    cs = ax.contour(X, Y, NPV_grid, levels=[0], linewidths=1.5)
    ax.clabel(cs, fmt={0: 'NPV = 0'})
except Exception:
    pass

plt.tight_layout()
plt.show()

# -----------------------------
# Quick sanity check at base case (midpoints)
# -----------------------------
mid_r = discount_rates[len(discount_rates)//2]
mid_g = revenue_growths[len(revenue_growths)//2]
print(f"Base-case discount rate: {mid_r:.2%}, growth: {mid_g:.2%}")

# Compute scalar NPV for the base case
rev_path = rev_year1 * (1 + mid_g) ** (years - 1)
rev_prev = np.concatenate([[rev_year1 / (1 + mid_g)], rev_path[:-1]])
EBITDA_b = rev_path * ebitda_margin
Dep_b = capex_pct_of_revenue * rev_path
EBIT_b = EBITDA_b - Dep_b
Taxes_b = np.maximum(EBIT_b, 0.0) * tax_rate
NOPAT_b = EBIT_b - Taxes_b
CAPEX_b = capex_pct_of_revenue * rev_path
Delta_NWC_b = nwc_pct_of_revenue * (rev_path - rev_prev)
FCF_b = NOPAT_b + Dep_b - CAPEX_b - Delta_NWC_b
NPV_b = (FCF_b / (1 + mid_r) ** years).sum() - initial_investment
print(f"Base-case NPV: {NPV_b:,.0f}")
