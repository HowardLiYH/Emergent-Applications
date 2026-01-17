#!/usr/bin/env python3
"""
Phase 7B: Run all expert-recommended audits.
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import grangercausalitytests
import json
from pathlib import Path


def audit_causal(si, feature):
    """Causal inference audit."""
    df = pd.concat([si, feature], axis=1).dropna()
    df.columns = ['si', 'feature']

    if len(df) < 100:
        return {'error': 'Insufficient data'}

    results = {}

    # SI → Feature
    try:
        test = grangercausalitytests(df[['feature', 'si']], maxlag=24, verbose=False)
        p_values = [test[lag][0]['ssr_ftest'][1] for lag in range(1, 25)]
        results['si_causes_feature'] = bool(min(p_values) < 0.05)
        results['si_causes_feature_p'] = float(min(p_values))
    except Exception as e:
        results['si_causes_feature'] = None
        results['si_causes_feature_error'] = str(e)

    # Feature → SI
    try:
        test = grangercausalitytests(df[['si', 'feature']], maxlag=24, verbose=False)
        p_values = [test[lag][0]['ssr_ftest'][1] for lag in range(1, 25)]
        results['feature_causes_si'] = bool(min(p_values) < 0.05)
        results['feature_causes_si_p'] = float(min(p_values))
    except Exception as e:
        results['feature_causes_si'] = None
        results['feature_causes_si_error'] = str(e)

    # Placebo: random SI
    real_r, _ = spearmanr(si, feature)
    placebo_rs = []
    for _ in range(100):
        fake_si = pd.Series(np.random.randn(len(si)), index=si.index)
        r, _ = spearmanr(fake_si.loc[feature.dropna().index], feature.dropna())
        placebo_rs.append(r)

    results['placebo_95_ci'] = list(np.percentile(placebo_rs, [2.5, 97.5]))
    results['real_exceeds_placebo'] = bool(abs(real_r) > np.percentile(np.abs(placebo_rs), 95))

    return results


def audit_permutation(si, feature, n_perm=1000):
    """Permutation test."""
    mask = ~(si.isna() | feature.isna())
    si_clean = si[mask].values
    feat_clean = feature[mask].values

    if len(si_clean) < 50:
        return {'error': 'Insufficient data'}

    real_r, _ = spearmanr(si_clean, feat_clean)

    shuffled_rs = []
    for _ in range(n_perm):
        shuffled_si = np.random.permutation(si_clean)
        r, _ = spearmanr(shuffled_si, feat_clean)
        shuffled_rs.append(r)

    p = np.mean(np.abs(shuffled_rs) >= np.abs(real_r))

    return {
        'real_r': float(real_r),
        'permutation_p': float(p),
        'significant': bool(p < 0.05)
    }


def audit_crypto(si, features, feature_name):
    """Crypto-specific audit: time-of-day, weekend effects."""
    if feature_name not in features.columns:
        return {'error': 'Feature not found'}

    feature = features[feature_name]

    # Align
    mask = ~(si.isna() | feature.isna())
    si_clean = si[mask]
    feat_clean = feature[mask]

    if len(si_clean) < 100:
        return {'error': 'Insufficient data'}

    results = {}

    # Time of day
    hour = si_clean.index.hour
    for session, (start, end) in [('asian', (0, 8)), ('eu', (8, 16)), ('us', (16, 24))]:
        session_mask = (hour >= start) & (hour < end)
        if session_mask.sum() > 50:
            r, p = spearmanr(si_clean[session_mask], feat_clean[session_mask])
            results[f'{session}_r'] = float(r)
            results[f'{session}_p'] = float(p)

    # Weekend vs weekday
    is_weekend = si_clean.index.dayofweek >= 5
    if (~is_weekend).sum() > 50 and is_weekend.sum() > 20:
        r_weekday, _ = spearmanr(si_clean[~is_weekend], feat_clean[~is_weekend])
        r_weekend, _ = spearmanr(si_clean[is_weekend], feat_clean[is_weekend])
        results['weekday_r'] = float(r_weekday)
        results['weekend_r'] = float(r_weekend)
        results['consistent'] = bool(np.sign(r_weekday) == np.sign(r_weekend))

    return results


def main():
    print("=" * 60)
    print("PHASE 7B: AUDITS")
    print("Running causal, permutation, and crypto audits")
    print("=" * 60)

    output_dir = Path("results/si_correlations")

    # Load metadata
    with open(output_dir / "metadata.json") as f:
        metadata = json.load(f)

    symbols = metadata.get('symbols', ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])

    # Load validation results to get confirmed features
    validation_file = output_dir / "validation_results.json"
    if validation_file.exists():
        with open(validation_file) as f:
            validation = json.load(f)

        # Get confirmed features
        confirmed = []
        for symbol, res in validation.items():
            confirmed.extend(res.get('confirmed', []))
        confirmed = list(set(confirmed))
    else:
        # Fallback: use significant features from discovery
        confirmed = []
        for symbol in symbols:
            disc_file = output_dir / f"discovery_{symbol}.csv"
            if disc_file.exists():
                df = pd.read_csv(disc_file)
                sig = df[df['significant'] == True]['feature'].tolist()
                confirmed.extend(sig)
        confirmed = list(set(confirmed))

    if not confirmed:
        confirmed = ['volatility_7d', 'trend_strength_7d']  # Fallback
        print(f"   No confirmed features, using fallback: {confirmed}")
    else:
        print(f"\n1. Features to audit: {confirmed}")

    all_audit_results = {}

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"AUDITING: {symbol}")
        print("=" * 60)

        # Load SI and features
        si_file = output_dir / f"si_{symbol}_train.csv"
        feat_file = output_dir / f"features_{symbol}_train.csv"

        if not si_file.exists() or not feat_file.exists():
            print(f"   Missing files for {symbol}")
            continue

        si = pd.read_csv(si_file, index_col=0, parse_dates=True).squeeze()
        features = pd.read_csv(feat_file, index_col=0, parse_dates=True)

        audit_results = {}

        for feature_name in confirmed[:5]:  # Top 5 only
            if feature_name not in features.columns:
                continue

            print(f"\n   Auditing: {feature_name}")
            feature = features[feature_name]

            # Align
            mask = ~(si.isna() | feature.isna())
            si_clean = si[mask]
            feat_clean = feature[mask]

            # Causal audit
            print("      Running causal audit...")
            causal = audit_causal(si_clean, feat_clean)
            audit_results[f'{feature_name}_causal'] = causal

            if 'si_causes_feature' in causal:
                status = "✅" if causal['si_causes_feature'] else "❌"
                print(f"      SI → {feature_name}: {status}")
            if 'feature_causes_si' in causal:
                status = "✅" if causal['feature_causes_si'] else "❌"
                print(f"      {feature_name} → SI: {status}")

            # Permutation audit
            print("      Running permutation audit...")
            perm = audit_permutation(si_clean, feat_clean)
            audit_results[f'{feature_name}_permutation'] = perm

            if 'significant' in perm:
                status = "✅ PASS" if perm['significant'] else "❌ FAIL"
                print(f"      Permutation test: {status} (p={perm.get('permutation_p', 'N/A'):.4f})")

            # Crypto audit
            print("      Running crypto audit...")
            crypto = audit_crypto(si, features, feature_name)
            audit_results[f'{feature_name}_crypto'] = crypto

            if 'consistent' in crypto:
                status = "✅" if crypto['consistent'] else "⚠️"
                print(f"      Weekend consistency: {status}")

        all_audit_results[symbol] = audit_results

    # Save
    print("\n" + "=" * 60)
    print("SAVING AUDIT RESULTS")
    print("=" * 60)

    with open(output_dir / "audit_results.json", "w") as f:
        json.dump(all_audit_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    print(f"\n✅ AUDITS COMPLETE!")
    print(f"Results saved to: {output_dir / 'audit_results.json'}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
