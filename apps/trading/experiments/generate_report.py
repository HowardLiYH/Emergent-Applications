#!/usr/bin/env python3
"""
Generate final report from corrected analysis.
Updated to use corrected_analysis format.
"""
import sys
sys.path.insert(0, '.')

import json
from pathlib import Path
from datetime import datetime


def main():
    print("=" * 70)
    print("FINAL REPORT: SI SIGNAL DISCOVERY")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load corrected analysis results
    try:
        with open("results/corrected_analysis/full_results.json") as f:
            full_results = json.load(f)
    except FileNotFoundError:
        print("❌ No corrected_analysis results found. Run run_corrected_analysis.py first.")
        return

    correlations = full_results.get('correlations', [])
    asset_results = full_results.get('results', {})
    config = full_results.get('config', {})

    # Load regime analysis if available
    regime_results = None
    try:
        with open("results/regime_analysis/regime_results.json") as f:
            regime_results = json.load(f)
    except:
        pass

    # ===== SECTION 1: Configuration =====
    print("\n" + "=" * 70)
    print("1. CONFIGURATION")
    print("=" * 70)
    print(f"   SI Windows: {config.get('si_windows', 'N/A')}")
    print(f"   N Agents per Strategy: {config.get('n_agents_options', 'N/A')}")
    print(f"   Purge Days: {config.get('purge_days', 'N/A')}")
    print(f"   Embargo Days: {config.get('embargo_days', 'N/A')}")
    print(f"   Effect Size Threshold: |r| > {config.get('effect_size_threshold', 0.1)}")

    # ===== SECTION 2: Discovery Results =====
    print("\n" + "=" * 70)
    print("2. DISCOVERY RESULTS")
    print("=" * 70)

    total_corr = len(correlations)
    meaningful = [c for c in correlations if c.get('globally_meaningful')]
    n_meaningful = len(meaningful)

    print(f"   Total correlations tested: {total_corr}")
    print(f"   Meaningful (FDR + effect size): {n_meaningful}")

    # Top correlates
    if meaningful:
        print("\n   Top 10 correlates:")
        sorted_corr = sorted(meaningful, key=lambda x: abs(x['r']), reverse=True)[:10]
        for c in sorted_corr:
            print(f"      {c['feature']:25} | r={c['r']:+.3f} | {c['symbol']} ({c['market']})")

    # ===== SECTION 3: Cross-Market Summary =====
    print("\n" + "=" * 70)
    print("3. CROSS-MARKET VALIDATION")
    print("=" * 70)

    market_summary = {}
    for market, assets in asset_results.items():
        market_summary[market] = {
            'n_assets': 0,
            'successful': 0,
            'val_rates': [],
            'test_rates': []
        }
        for symbol, data in assets.items():
            market_summary[market]['n_assets'] += 1
            if isinstance(data, dict) and data.get('status') == 'success':
                market_summary[market]['successful'] += 1
                market_summary[market]['val_rates'].append(data.get('val_confirmation_rate', 0))
                market_summary[market]['test_rates'].append(data.get('test_confirmation_rate', 0))

    print("\n   Market | Assets | Val Rate | Test Rate")
    print("   " + "-" * 50)

    import numpy as np
    for market, stats in market_summary.items():
        n = stats['n_assets']
        success = stats['successful']
        val_rate = np.mean(stats['val_rates']) * 100 if stats['val_rates'] else 0
        test_rate = np.mean(stats['test_rates']) * 100 if stats['test_rates'] else 0
        print(f"   {market:12} | {success}/{n} | {val_rate:5.1f}% | {test_rate:5.1f}%")

    # Overall
    all_val = []
    all_test = []
    for stats in market_summary.values():
        all_val.extend(stats['val_rates'])
        all_test.extend(stats['test_rates'])

    print("   " + "-" * 50)
    print(f"   {'OVERALL':12} | {sum(s['successful'] for s in market_summary.values())}/{sum(s['n_assets'] for s in market_summary.values())} | {np.mean(all_val)*100:.1f}% | {np.mean(all_test)*100:.1f}%")

    # ===== SECTION 4: Feature Consistency =====
    print("\n" + "=" * 70)
    print("4. FEATURE CONSISTENCY ACROSS MARKETS")
    print("=" * 70)

    # Count how many markets each feature appears in
    from collections import defaultdict
    feature_markets = defaultdict(set)
    feature_signs = defaultdict(list)

    for c in meaningful:
        feature_markets[c['feature']].add(c['market'])
        feature_signs[c['feature']].append(np.sign(c['r']))

    # Features in 2+ markets
    multi_market_features = {f: markets for f, markets in feature_markets.items() if len(markets) >= 2}

    print(f"\n   Features significant in 2+ markets: {len(multi_market_features)}")

    for feat in sorted(multi_market_features.keys(), key=lambda x: -len(multi_market_features[x])):
        markets = multi_market_features[feat]
        signs = feature_signs[feat]
        consistent = len(set(signs)) == 1
        consistency_str = "✅" if consistent else "⚠️"
        print(f"      {feat:25} | {len(markets)} markets | {consistency_str}")

    # ===== SECTION 5: Regime Analysis =====
    if regime_results:
        print("\n" + "=" * 70)
        print("5. REGIME-CONDITIONED ANALYSIS")
        print("=" * 70)

        total_flips = regime_results.get('total_flips', 0)
        total_tested = regime_results.get('total_features', 0)

        print(f"\n   Sign flips across regimes: {total_flips}/{total_tested} ({total_flips/total_tested*100:.1f}%)")

        if total_flips > 0:
            print("\n   Features with regime-dependent correlations:")
            for flip in regime_results.get('sign_flips_summary', [])[:10]:
                print(f"      {flip['feature']:25} | {flip['asset']} | {flip['signs']}")

    # ===== SECTION 6: Exit Criteria Check =====
    print("\n" + "=" * 70)
    print("6. EXIT CRITERIA CHECK")
    print("=" * 70)

    # Criterion 1: ≥3 features with |r| > 0.15
    strong_features = set(c['feature'] for c in meaningful if abs(c['r']) > 0.15)
    criterion_1 = len(strong_features) >= 3

    # Criterion 2: Confirmed in both val and test
    avg_val = np.mean(all_val) if all_val else 0
    avg_test = np.mean(all_test) if all_test else 0
    criterion_2 = avg_val > 0.3 and avg_test > 0.2

    # Criterion 3: Replicate in ≥3 assets
    assets_with_findings = set(c['symbol'] for c in meaningful)
    criterion_3 = len(assets_with_findings) >= 3

    # Criterion 4: Replicate in ≥2 markets
    markets_with_findings = set(c['market'] for c in meaningful)
    criterion_4 = len(markets_with_findings) >= 2

    print(f"\n   [{'✅' if criterion_1 else '❌'}] ≥3 features with |r| > 0.15: {len(strong_features)} features")
    print(f"   [{'✅' if criterion_2 else '❌'}] Confirmed in VAL ({avg_val*100:.0f}%) and TEST ({avg_test*100:.0f}%)")
    print(f"   [{'✅' if criterion_3 else '❌'}] Replicate in ≥3 assets: {len(assets_with_findings)} assets")
    print(f"   [{'✅' if criterion_4 else '❌'}] Replicate in ≥2 markets: {len(markets_with_findings)} markets")

    # ===== SECTION 7: Conclusion =====
    print("\n" + "=" * 70)
    print("7. CONCLUSION")
    print("=" * 70)

    all_criteria = criterion_1 and criterion_2 and criterion_3 and criterion_4

    if all_criteria:
        print("\n   ✅ PRIMARY HYPOTHESIS SUPPORTED")
        print(f"      - SI correlates with {len(strong_features)} features (|r| > 0.15)")
        print(f"      - Findings replicate across {len(assets_with_findings)} assets, {len(markets_with_findings)} markets")
        print(f"      - VAL: {avg_val*100:.0f}%, TEST: {avg_test*100:.0f}% confirmation rate")
    else:
        print("\n   ⚠️ PARTIAL SUCCESS")
        print("      - Some criteria not fully met")

    # Key findings
    print("\n   Key SI Correlates (replicate across markets):")
    for feat in list(multi_market_features.keys())[:5]:
        print(f"      • {feat}")

    # Caveats
    print("\n   ⚠️ Caveats:")
    if regime_results and regime_results.get('total_flips', 0) > 0:
        flip_pct = regime_results['total_flips'] / regime_results['total_features'] * 100
        print(f"      • {flip_pct:.0f}% of correlations flip sign across regimes")
    print("      • Simple strategies used (not production-grade)")
    print("      • No market impact or slippage modeled")

    # ===== SECTION 8: Next Steps =====
    print("\n" + "=" * 70)
    print("8. NEXT STEPS")
    print("=" * 70)

    if all_criteria:
        print("\n   Recommended next steps:")
        print("      1. ✅ SI signal validated → Proceed to paper writing")
        print("      2. Consider regime-specific SI strategies")
        print("      3. Test SI-based trading strategy with transaction costs")
        print("      4. Explore regime prediction (future/REGIME_PREDICTION.md)")
    else:
        print("\n   Recommended next steps:")
        print("      1. Investigate failed criteria")
        print("      2. Try alternative SI configurations")
        print("      3. Check data quality issues")

    # Save report
    report = {
        'generated': datetime.now().isoformat(),
        'config': config,
        'discovery': {
            'total_correlations': total_corr,
            'meaningful': n_meaningful,
            'top_features': [c['feature'] for c in sorted_corr][:10] if meaningful else []
        },
        'cross_market': {
            'markets_tested': list(market_summary.keys()),
            'avg_val_rate': float(np.mean(all_val)) if all_val else 0,
            'avg_test_rate': float(np.mean(all_test)) if all_test else 0
        },
        'exit_criteria': {
            'criterion_1_features': bool(criterion_1),
            'criterion_2_validation': bool(criterion_2),
            'criterion_3_assets': bool(criterion_3),
            'criterion_4_markets': bool(criterion_4),
            'all_met': bool(all_criteria)
        },
        'conclusion': 'SUCCESS' if all_criteria else 'PARTIAL'
    }

    output_path = Path("results/si_correlations/final_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Report saved to {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
