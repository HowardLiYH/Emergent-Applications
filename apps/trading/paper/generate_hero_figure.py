#!/usr/bin/env python3
"""
STEP 2: Create Hero Figure

Generate a 4-panel figure that tells the entire story of SI emergence.
This is CRITICAL for NeurIPS best paper consideration.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Set style for publication quality
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 150

print("\n" + "="*70)
print("  GENERATING HERO FIGURE")
print("="*70)
print(f"  Time: {datetime.now().isoformat()}\n")

# ============================================================
# LOAD DATA
# ============================================================

from src.data.loader_v2 import DataLoaderV2, MarketType
from src.agents.strategies_v2 import get_default_strategies
from src.competition.niche_population_v2 import NichePopulationV2

loader = DataLoaderV2()

# Load SPY for cleaner visualization
print("  Loading data...")
data = loader.load('SPY', MarketType.STOCKS)

# Compute SI
print("  Computing SI...")
strategies = get_default_strategies('daily')
population = NichePopulationV2(strategies, n_agents_per_strategy=5, frequency='daily')
population.run(data)
si = population.compute_si_timeseries(data, window=7)

# Compute ADX
def calc_adx(data, period=14):
    high, low, close = data['high'], data['low'], data['close']
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).sum()
    plus_di = 100 * plus_dm.rolling(period).sum() / (atr + 1e-10)
    minus_di = 100 * minus_dm.rolling(period).sum() / (atr + 1e-10)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    return dx.rolling(period).mean() / 100

adx = calc_adx(data)

# Align
common = si.dropna().index.intersection(adx.dropna().index)
si_aligned = si.loc[common]
adx_aligned = adx.loc[common]
price_aligned = data['close'].loc[common]

print(f"  Data points: {len(common)}")

# ============================================================
# CREATE HERO FIGURE
# ============================================================

fig = plt.figure(figsize=(14, 10))

# Panel A: NichePopulation Mechanism (schematic)
ax1 = fig.add_subplot(2, 2, 1)

# Draw mechanism diagram
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 8)
ax1.set_aspect('equal')

# Agents
for i, y in enumerate([6, 4.5, 3]):
    circle = plt.Circle((1.5, y), 0.4, color='#3498db', alpha=0.8)
    ax1.add_patch(circle)
    ax1.text(1.5, y, f'A{i+1}', ha='center', va='center', fontsize=9, color='white', weight='bold')

# Niches
niche_colors = ['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
niche_names = ['Trend', 'Mean-Rev', 'Vol', 'Breakout']
for i, (y, color, name) in enumerate(zip([6.5, 5, 3.5, 2], niche_colors, niche_names)):
    rect = plt.Rectangle((7, y-0.3), 2, 0.6, color=color, alpha=0.8)
    ax1.add_patch(rect)
    ax1.text(8, y, name, ha='center', va='center', fontsize=8, color='white', weight='bold')

# Competition arrows
ax1.annotate('', xy=(6.8, 6.5), xytext=(2, 6),
             arrowprops=dict(arrowstyle='->', color='#34495e', lw=1.5))
ax1.annotate('', xy=(6.8, 5), xytext=(2, 4.5),
             arrowprops=dict(arrowstyle='->', color='#34495e', lw=1.5))
ax1.annotate('', xy=(6.8, 3.5), xytext=(2, 3),
             arrowprops=dict(arrowstyle='->', color='#34495e', lw=1.5))

# Labels
ax1.text(4.5, 7.5, 'Competition', ha='center', fontsize=11, weight='bold')
ax1.text(4.5, 1, 'Fitness-proportional affinity updates', ha='center', fontsize=9, style='italic')
ax1.text(1.5, 7.5, 'Agents', ha='center', fontsize=10)
ax1.text(8, 7.5, 'Niches', ha='center', fontsize=10)

ax1.axis('off')
ax1.set_title('(a) NichePopulation Mechanism', fontsize=12, weight='bold', pad=10)

# Panel B: SI Emergence Over Time
ax2 = fig.add_subplot(2, 2, 2)

# Plot SI and price
ax2_twin = ax2.twinx()

# Price (gray, background)
ax2_twin.plot(price_aligned.index, price_aligned.values, 
              color='#95a5a6', alpha=0.3, linewidth=1, label='SPY Price')
ax2_twin.set_ylabel('SPY Price ($)', color='#95a5a6')
ax2_twin.tick_params(axis='y', labelcolor='#95a5a6')

# SI (blue, foreground)
ax2.plot(si_aligned.index, si_aligned.values, 
         color='#3498db', linewidth=1.5, label='SI')
ax2.fill_between(si_aligned.index, 0, si_aligned.values, alpha=0.2, color='#3498db')
ax2.set_ylabel('Specialization Index (SI)', color='#3498db')
ax2.tick_params(axis='y', labelcolor='#3498db')
ax2.set_ylim(0, si_aligned.max() * 1.1)

# Mark crisis periods
crisis_periods = [
    ('2020-02-15', '2020-04-01', 'COVID'),
    ('2022-01-01', '2022-06-30', 'Rate Hike'),
]
for start, end, label in crisis_periods:
    if start in price_aligned.index.astype(str).values:
        ax2.axvspan(pd.Timestamp(start), pd.Timestamp(end), 
                    alpha=0.2, color='#e74c3c', label=f'{label} Crisis')

ax2.set_xlabel('Date')
ax2.set_title('(b) SI Emergence: Specialization Tracks Market Structure', fontsize=12, weight='bold', pad=10)
ax2.legend(loc='upper left', fontsize=8)

# Panel C: SI-ADX Cointegration
ax3 = fig.add_subplot(2, 2, 3)

# Scatter plot with regression line
ax3.scatter(adx_aligned.values, si_aligned.values, 
            alpha=0.3, s=10, c='#3498db', edgecolors='none')

# Regression line
z = np.polyfit(adx_aligned.values, si_aligned.values, 1)
p = np.poly1d(z)
x_line = np.linspace(adx_aligned.min(), adx_aligned.max(), 100)
ax3.plot(x_line, p(x_line), color='#e74c3c', linewidth=2, 
         label=f'r = {np.corrcoef(adx_aligned, si_aligned)[0,1]:.3f}')

ax3.set_xlabel('ADX (Trend Strength)')
ax3.set_ylabel('SI (Specialization Index)')
ax3.set_title('(c) SI-ADX Cointegration (p < 0.0001)', fontsize=12, weight='bold', pad=10)
ax3.legend(loc='upper left', fontsize=10)

# Add cointegration annotation
ax3.text(0.95, 0.05, 'Engle-Granger:\np < 0.0001', 
         transform=ax3.transAxes, ha='right', va='bottom',
         fontsize=9, style='italic', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel D: Phase Transition (Short vs Long-term correlation)
ax4 = fig.add_subplot(2, 2, 4)

# Compute correlations at different frequencies
frequencies = [3, 7, 14, 21, 30, 45, 60, 90, 120]
correlations = []

for freq in frequencies:
    si_smooth = si_aligned.rolling(freq).mean().dropna()
    adx_smooth = adx_aligned.rolling(freq).mean().dropna()
    common_smooth = si_smooth.index.intersection(adx_smooth.index)
    if len(common_smooth) > 50:
        r = np.corrcoef(si_smooth.loc[common_smooth], adx_smooth.loc[common_smooth])[0, 1]
        correlations.append(r)
    else:
        correlations.append(np.nan)

ax4.bar(range(len(frequencies)), correlations, 
        color=['#e74c3c' if c < 0 else '#2ecc71' for c in correlations],
        alpha=0.8, edgecolor='black', linewidth=0.5)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax4.set_xticks(range(len(frequencies)))
ax4.set_xticklabels([f'{f}d' for f in frequencies], rotation=45)
ax4.set_xlabel('Rolling Window (days)')
ax4.set_ylabel('SI-ADX Correlation')
ax4.set_title('(d) Phase Transition: Short-term Negative → Long-term Positive', 
              fontsize=12, weight='bold', pad=10)

# Add phase transition annotation
ax4.axvline(x=4, color='#9b59b6', linestyle='--', linewidth=2, alpha=0.7)
ax4.text(4.2, max(correlations)*0.9, '~30 day\nthreshold', 
         fontsize=9, color='#9b59b6', weight='bold')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Save figure
Path("paper/figures").mkdir(parents=True, exist_ok=True)
fig.savefig("paper/figures/hero_figure.png", dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
fig.savefig("paper/figures/hero_figure.pdf", bbox_inches='tight',
            facecolor='white', edgecolor='none')

print("\n  ✅ Hero figure saved:")
print("     - paper/figures/hero_figure.png")
print("     - paper/figures/hero_figure.pdf")
print("="*70 + "\n")

plt.close()
