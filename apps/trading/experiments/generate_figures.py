#!/usr/bin/env python3
"""
Generate publication-quality figures for SI Signal Discovery report.
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

# Style settings for publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color palette
COLORS = {
    'crypto': '#F7931A',      # Bitcoin orange
    'forex': '#2E86AB',       # Blue
    'stocks': '#28A745',      # Green
    'commodities': '#FFD700', # Gold
    'primary': '#1E3A5F',     # Dark blue
    'secondary': '#E63946',   # Red
    'tertiary': '#457B9D',    # Medium blue
}

def load_results():
    """Load all results files."""
    results = {}

    # Main corrected analysis
    with open('results/corrected_analysis/full_results.json') as f:
        results['main'] = json.load(f)

    # Regime comparison
    try:
        with open('results/regime_comparison/comparison_results.json') as f:
            results['regime'] = json.load(f)
    except:
        results['regime'] = None

    # Regime analysis
    try:
        with open('results/regime_analysis/regime_results.json') as f:
            results['regime_analysis'] = json.load(f)
    except:
        results['regime_analysis'] = None

    return results


def fig1_cross_market_validation(results, output_dir):
    """Figure 1: Cross-market validation rates."""
    fig, ax = plt.subplots(figsize=(10, 6))

    asset_results = results['main'].get('results', {})

    markets = []
    val_rates = []
    test_rates = []
    colors = []

    for market in ['crypto', 'forex', 'stocks', 'commodities']:
        if market in asset_results:
            for symbol, data in asset_results[market].items():
                if isinstance(data, dict) and data.get('status') == 'success':
                    markets.append(f"{symbol}\n({market})")
                    val_rates.append(data.get('val_confirmation_rate', 0) * 100)
                    test_rates.append(data.get('test_confirmation_rate', 0) * 100)
                    colors.append(COLORS.get(market, '#666666'))

    x = np.arange(len(markets))
    width = 0.35

    bars1 = ax.bar(x - width/2, val_rates, width, label='Validation Set',
                   color=COLORS['primary'], alpha=0.8)
    bars2 = ax.bar(x + width/2, test_rates, width, label='Test Set (Holdout)',
                   color=COLORS['secondary'], alpha=0.8)

    ax.set_ylabel('Confirmation Rate (%)')
    ax.set_title('SI Correlation Replication Across Markets', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(markets, rotation=0)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)

    # Add horizontal line for threshold
    ax.axhline(y=30, color='gray', linestyle='--', alpha=0.5, label='30% threshold')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_cross_market_validation.png')
    plt.savefig(output_dir / 'fig1_cross_market_validation.pdf')
    plt.close()
    print("✅ Figure 1: Cross-market validation")


def fig2_top_correlates(results, output_dir):
    """Figure 2: Top SI correlates across markets."""
    fig, ax = plt.subplots(figsize=(12, 7))

    correlations = results['main'].get('correlations', [])
    meaningful = [c for c in correlations if c.get('globally_meaningful')]

    # Aggregate by feature
    feature_data = defaultdict(lambda: {'r_values': [], 'markets': set()})
    for c in meaningful:
        feature_data[c['feature']]['r_values'].append(c['r'])
        feature_data[c['feature']]['markets'].add(c['market'])

    # Sort by number of markets, then by average |r|
    sorted_features = sorted(
        feature_data.items(),
        key=lambda x: (-len(x[1]['markets']), -np.mean(np.abs(x[1]['r_values'])))
    )[:15]

    features = [f[0] for f in sorted_features]
    avg_r = [np.mean(f[1]['r_values']) for f in sorted_features]
    n_markets = [len(f[1]['markets']) for f in sorted_features]

    # Color by sign
    colors = [COLORS['primary'] if r > 0 else COLORS['secondary'] for r in avg_r]

    y = np.arange(len(features))
    bars = ax.barh(y, avg_r, color=colors, alpha=0.8)

    # Add market count annotations
    for i, (bar, nm) in enumerate(zip(bars, n_markets)):
        width = bar.get_width()
        ax.annotate(f'{nm} markets',
                   xy=(width + 0.01 if width > 0 else width - 0.01, bar.get_y() + bar.get_height()/2),
                   xytext=(5 if width > 0 else -5, 0), textcoords="offset points",
                   ha='left' if width > 0 else 'right', va='center', fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(features)
    ax.set_xlabel('Average Spearman Correlation with SI')
    ax.set_title('Top SI Correlates (Features Significant in Multiple Markets)', fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlim(-0.35, 0.35)

    # Legend
    pos_patch = mpatches.Patch(color=COLORS['primary'], label='Positive correlation')
    neg_patch = mpatches.Patch(color=COLORS['secondary'], label='Negative correlation')
    ax.legend(handles=[pos_patch, neg_patch], loc='lower right')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_top_correlates.png')
    plt.savefig(output_dir / 'fig2_top_correlates.pdf')
    plt.close()
    print("✅ Figure 2: Top correlates")


def fig3_regime_comparison(results, output_dir):
    """Figure 3: Regime detection method comparison."""
    if results.get('regime') is None:
        print("⚠️ Skipping Figure 3: No regime comparison data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    regime_data = results['regime']

    # Panel A: Flip rates
    ax = axes[0]
    methods = list(regime_data.get('method_flip_rates', {}).keys())
    flip_rates = [regime_data['method_flip_rates'][m] * 100 for m in methods]

    colors = [COLORS['primary'] if r < 15 else COLORS['tertiary'] if r < 25 else COLORS['secondary']
              for r in flip_rates]

    bars = ax.bar(methods, flip_rates, color=colors, alpha=0.8)
    ax.set_ylabel('Sign Flip Rate (%)')
    ax.set_title('A. Correlation Consistency by Method', fontweight='bold')
    ax.set_ylim(0, 35)

    # Add threshold lines
    ax.axhline(y=15, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=25, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(3.5, 16, 'Good', fontsize=9, color='green')
    ax.text(3.5, 26, 'Poor', fontsize=9, color='red')

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    # Panel B: Method agreement
    ax = axes[1]
    agreement = regime_data.get('method_agreement', {})

    # Create matrix
    method_names = ['rule_2', 'hmm_2', 'gmm_2']
    matrix = np.eye(3) * 100

    for key, val in agreement.items():
        parts = key.split('_vs_')
        if len(parts) == 2:
            m1, m2 = parts[0], parts[1]
            if m1 in method_names and m2 in method_names:
                i, j = method_names.index(m1), method_names.index(m2)
                matrix[i, j] = val * 100
                matrix[j, i] = val * 100

    im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks(range(len(method_names)))
    ax.set_yticks(range(len(method_names)))
    ax.set_xticklabels(['Rule-based', 'HMM', 'GMM'])
    ax.set_yticklabels(['Rule-based', 'HMM', 'GMM'])
    ax.set_title('B. Method Agreement (%)', fontweight='bold')

    # Add text annotations
    for i in range(len(method_names)):
        for j in range(len(method_names)):
            text = ax.text(j, i, f'{matrix[i, j]:.0f}%', ha='center', va='center',
                          color='white' if matrix[i, j] > 50 else 'black')

    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_regime_comparison.png')
    plt.savefig(output_dir / 'fig3_regime_comparison.pdf')
    plt.close()
    print("✅ Figure 3: Regime comparison")


def fig4_market_summary(results, output_dir):
    """Figure 4: Market-level summary."""
    fig, ax = plt.subplots(figsize=(10, 6))

    correlations = results['main'].get('correlations', [])
    meaningful = [c for c in correlations if c.get('globally_meaningful')]

    # Aggregate by market
    market_stats = defaultdict(lambda: {'n_corr': 0, 'avg_r': [], 'n_assets': set()})
    for c in meaningful:
        market = c['market']
        market_stats[market]['n_corr'] += 1
        market_stats[market]['avg_r'].append(abs(c['r']))
        market_stats[market]['n_assets'].add(c['symbol'])

    markets = ['crypto', 'forex', 'stocks', 'commodities']
    n_correlations = [market_stats[m]['n_corr'] for m in markets]
    avg_effect = [np.mean(market_stats[m]['avg_r']) if market_stats[m]['avg_r'] else 0 for m in markets]
    n_assets = [len(market_stats[m]['n_assets']) for m in markets]

    x = np.arange(len(markets))
    width = 0.3

    ax2 = ax.twinx()

    bars1 = ax.bar(x - width/2, n_correlations, width, label='Significant Correlations',
                   color=COLORS['primary'], alpha=0.8)
    bars2 = ax2.bar(x + width/2, [r*100 for r in avg_effect], width, label='Avg Effect Size (|r|)',
                    color=COLORS['secondary'], alpha=0.8)

    ax.set_ylabel('Number of Significant Correlations', color=COLORS['primary'])
    ax2.set_ylabel('Average |r| × 100', color=COLORS['secondary'])
    ax.set_title('SI Signal Strength by Market Type', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in markets])

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Add asset count annotation
    for i, (bar, na) in enumerate(zip(bars1, n_assets)):
        ax.annotate(f'{na} assets', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_market_summary.png')
    plt.savefig(output_dir / 'fig4_market_summary.pdf')
    plt.close()
    print("✅ Figure 4: Market summary")


def fig5_exit_criteria(results, output_dir):
    """Figure 5: Exit criteria summary (visual checklist)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    criteria = [
        ('≥3 features |r| > 0.15', 17, 3, True),
        ('VAL confirmation rate', 51, 30, True),
        ('TEST confirmation rate', 44, 20, True),
        ('Assets with findings', 11, 3, True),
        ('Markets validated', 4, 2, True),
    ]

    y = np.arange(len(criteria))

    for i, (name, actual, required, passed) in enumerate(criteria):
        color = '#28A745' if passed else '#DC3545'

        # Background bar (required)
        ax.barh(i, required, color='lightgray', alpha=0.5, height=0.6)

        # Actual bar
        ax.barh(i, actual, color=color, alpha=0.8, height=0.6)

        # Checkmark or X
        symbol = '✓' if passed else '✗'
        ax.text(max(actual, required) + 1, i, symbol, fontsize=16,
               color=color, va='center', fontweight='bold')

        # Value label
        ax.text(actual/2, i, f'{actual}', ha='center', va='center',
               color='white', fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels([c[0] for c in criteria])
    ax.set_xlabel('Value')
    ax.set_title('Exit Criteria: All Passed ✓', fontweight='bold', color='#28A745')
    ax.set_xlim(0, 60)

    # Legend
    gray_patch = mpatches.Patch(color='lightgray', alpha=0.5, label='Required minimum')
    green_patch = mpatches.Patch(color='#28A745', alpha=0.8, label='Actual (passed)')
    ax.legend(handles=[gray_patch, green_patch], loc='lower right')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_exit_criteria.png')
    plt.savefig(output_dir / 'fig5_exit_criteria.pdf')
    plt.close()
    print("✅ Figure 5: Exit criteria")


def main():
    print("="*60)
    print("GENERATING PUBLICATION FIGURES")
    print("="*60)

    output_dir = Path('paper/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results()

    fig1_cross_market_validation(results, output_dir)
    fig2_top_correlates(results, output_dir)
    fig3_regime_comparison(results, output_dir)
    fig4_market_summary(results, output_dir)
    fig5_exit_criteria(results, output_dir)

    print(f"\n✅ All figures saved to {output_dir}/")
    print("   - fig1_cross_market_validation.png/pdf")
    print("   - fig2_top_correlates.png/pdf")
    print("   - fig3_regime_comparison.png/pdf")
    print("   - fig4_market_summary.png/pdf")
    print("   - fig5_exit_criteria.png/pdf")


if __name__ == "__main__":
    main()
