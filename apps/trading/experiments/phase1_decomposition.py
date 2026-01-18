#!/usr/bin/env python3
"""
PHASE 1 DECOMPOSITION ANALYSIS
- Wavelet Decomposition (multi-scale)
- Principal Component Analysis
- Independent Component Analysis
- Cross-Correlation at Multiple Lags
- Information Coefficient Decay
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
import pywt
import warnings
warnings.filterwarnings('ignore')

from src.agents.strategies_v2 import get_default_strategies
from src.competition.niche_population_v2 import NichePopulationV2
from src.data.loader_v2 import DataLoaderV2, MarketType

discoveries = []

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)

def discover(msg):
    discoveries.append(msg)
    print(f"  📍 {msg}")

def compute_si(data, window=7):
    strategies = get_default_strategies('daily')
    population = NichePopulationV2(strategies, n_agents_per_strategy=3, frequency='daily')
    population.run(data)
    return population.compute_si_timeseries(data, window=window)

def compute_features(data):
    features = pd.DataFrame(index=data.index)
    returns = data['close'].pct_change()
    
    # ADX
    high_low = data['high'] - data['low']
    high_close = abs(data['high'] - data['close'].shift())
    low_close = abs(data['low'] - data['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_dm = (data['high'] - data['high'].shift()).clip(lower=0)
    minus_dm = (data['low'].shift() - data['low']).clip(lower=0)
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr + 1e-10))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    features['adx'] = dx.rolling(14).mean()
    
    # RSI Extremity
    delta = data['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    features['rsi_extremity'] = abs(rsi - 50)
    
    # Volatility
    features['volatility'] = returns.rolling(14).std()
    
    # Returns
    features['returns'] = returns
    
    # Momentum
    features['momentum'] = data['close'].pct_change(7)
    
    # Volume ratio
    if 'volume' in data.columns and data['volume'].sum() > 0:
        features['vol_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
    
    return features.dropna()

def wavelet_decompose(signal, wavelet='db4', level=4):
    """
    Decompose signal into different frequency bands using wavelets.
    Returns: approximation (low freq) + details (high freq)
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Reconstruct each level
    reconstructions = {}
    
    for i in range(level + 1):
        coeffs_copy = [np.zeros_like(c) for c in coeffs]
        coeffs_copy[i] = coeffs[i]
        
        reconstructed = pywt.waverec(coeffs_copy, wavelet)
        # Trim to original length
        reconstructed = reconstructed[:len(signal)]
        
        if i == 0:
            reconstructions['approx'] = reconstructed  # Low frequency trend
        else:
            reconstructions[f'detail_{i}'] = reconstructed  # High frequency details
    
    return reconstructions

def main():
    print("\n" + "="*70)
    print("  PHASE 1 DECOMPOSITION ANALYSIS")
    print("="*70)
    print(f"  Time: {datetime.now().isoformat()}")
    
    loader = DataLoaderV2()
    
    assets = {
        'BTCUSDT': (loader.load('BTCUSDT', MarketType.CRYPTO), MarketType.CRYPTO),
        'SPY': (loader.load('SPY', MarketType.STOCKS), MarketType.STOCKS),
        'EURUSD': (loader.load('EURUSD', MarketType.FOREX), MarketType.FOREX),
    }
    
    all_results = {}
    
    # ============================================================
    section("1. WAVELET DECOMPOSITION")
    # ============================================================
    
    print("\n  Decomposing SI into frequency components")
    print("  Low freq = long-term trend, High freq = short-term noise")
    
    wavelet_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        features = compute_features(data)
        
        # Wavelet decomposition of SI
        si_clean = si.dropna()
        decomp = wavelet_decompose(si_clean.values, level=4)
        
        # Compute variance at each level
        total_var = np.var(si_clean.values)
        variances = {}
        for level_name, recon in decomp.items():
            var_pct = np.var(recon) / total_var * 100
            variances[level_name] = var_pct
        
        print(f"    Variance by frequency band:")
        print(f"      Low freq (trend):    {variances['approx']:.1f}%")
        for i in range(1, 5):
            if f'detail_{i}' in variances:
                print(f"      Detail level {i}:       {variances[f'detail_{i}']:.1f}%")
        
        # Correlate each component with ADX
        common = si.index.intersection(features.index)
        adx = features.loc[common, 'adx']
        
        component_corrs = {}
        for level_name, recon in decomp.items():
            # Align
            recon_series = pd.Series(recon, index=si_clean.index)
            aligned = recon_series.loc[common]
            if len(aligned) > 50:
                r, _ = spearmanr(aligned, adx.loc[aligned.index])
                component_corrs[level_name] = r
        
        print(f"\n    Correlation with ADX by frequency:")
        for level_name, r in component_corrs.items():
            print(f"      {level_name}: r = {r:+.3f}")
        
        wavelet_results[name] = {
            'variances': {k: float(v) for k, v in variances.items()},
            'correlations_with_adx': {k: float(v) for k, v in component_corrs.items()},
        }
        
        # Check if low freq dominates correlation
        if 'approx' in component_corrs:
            if abs(component_corrs['approx']) > abs(component_corrs.get('detail_1', 0)) * 1.5:
                discover(f"SI-ADX correlation is dominated by low-freq trend in {name}")
    
    all_results['wavelet'] = wavelet_results
    
    # ============================================================
    section("2. MULTI-LAG CROSS-CORRELATION")
    # ============================================================
    
    print("\n  Information Coefficient (IC) at different lags")
    
    ic_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        features = compute_features(data)
        
        common = si.index.intersection(features.index)
        si_aligned = si.loc[common]
        adx_aligned = features.loc[common, 'adx']
        returns_aligned = features.loc[common, 'returns']
        
        # IC with future returns at different lags
        lags = [1, 2, 3, 5, 7, 10, 14, 21]
        ic_decay = []
        
        print(f"    {'Lag':>5} {'IC(SI→Ret)':>12} {'IC(SI→ADX)':>12}")
        print("    " + "-"*32)
        
        for lag in lags:
            # SI predicting future returns
            future_ret = returns_aligned.shift(-lag)
            common_lag = si_aligned.dropna().index.intersection(future_ret.dropna().index)
            
            if len(common_lag) > 100:
                ic_ret, _ = spearmanr(si_aligned.loc[common_lag], future_ret.loc[common_lag])
                
                # SI vs lagged ADX
                adx_lagged = adx_aligned.shift(lag)
                common_adx = si_aligned.dropna().index.intersection(adx_lagged.dropna().index)
                if len(common_adx) > 100:
                    ic_adx, _ = spearmanr(si_aligned.loc[common_adx], adx_lagged.loc[common_adx])
                else:
                    ic_adx = np.nan
                
                ic_decay.append({
                    'lag': lag,
                    'ic_returns': float(ic_ret),
                    'ic_adx': float(ic_adx) if not np.isnan(ic_adx) else None,
                })
                
                ic_adx_str = f"{ic_adx:+.4f}" if not np.isnan(ic_adx) else "N/A"
                print(f"    {lag:>5}d {ic_ret:>+12.4f} {ic_adx_str:>12}")
        
        ic_results[name] = ic_decay
        
        # Compute IC half-life
        if ic_decay:
            ic_1 = abs(ic_decay[0]['ic_returns']) if ic_decay[0]['ic_returns'] else 0
            half_life = None
            for entry in ic_decay:
                if abs(entry['ic_returns']) < ic_1 / 2:
                    half_life = entry['lag']
                    break
            
            if half_life:
                discover(f"SI→Returns IC half-life = {half_life} days in {name}")
    
    all_results['ic_decay'] = ic_results
    
    # ============================================================
    section("3. PRINCIPAL COMPONENT ANALYSIS")
    # ============================================================
    
    print("\n  Is SI captured by first PC of market features?")
    
    pca_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        features = compute_features(data)
        
        common = si.index.intersection(features.index)
        si_aligned = si.loc[common]
        
        # Prepare feature matrix
        feature_cols = ['adx', 'rsi_extremity', 'volatility', 'momentum']
        if 'vol_ratio' in features.columns:
            feature_cols.append('vol_ratio')
        
        X = features.loc[common, feature_cols].dropna()
        common_final = X.index.intersection(si_aligned.dropna().index)
        X = X.loc[common_final]
        si_final = si_aligned.loc[common_final]
        
        if len(X) < 100:
            print("    Not enough data")
            continue
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        pca = PCA(n_components=min(3, len(feature_cols)))
        pcs = pca.fit_transform(X_scaled)
        
        print(f"    Variance explained: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")
        
        # Correlate SI with each PC
        pc_corrs = []
        for i in range(pcs.shape[1]):
            r, _ = spearmanr(si_final, pcs[:, i])
            pc_corrs.append(r)
            print(f"    SI vs PC{i+1}: r = {r:+.3f}")
        
        # Check loadings
        print(f"\n    PC1 loadings:")
        for j, col in enumerate(feature_cols):
            print(f"      {col}: {pca.components_[0, j]:+.3f}")
        
        pca_results[name] = {
            'variance_explained': pca.explained_variance_ratio_.tolist(),
            'si_pc_correlations': pc_corrs,
            'pc1_loadings': {col: float(pca.components_[0, j]) for j, col in enumerate(feature_cols)},
        }
        
        if abs(pc_corrs[0]) > 0.2:
            discover(f"SI strongly correlates with PC1 in {name} (r={pc_corrs[0]:.3f})")
    
    all_results['pca'] = pca_results
    
    # ============================================================
    section("4. FREQUENCY BAND CORRELATIONS")
    # ============================================================
    
    print("\n  At which frequency does SI-ADX relationship exist?")
    
    freq_results = {}
    
    def bandpass_filter(data, lowcut, highcut, fs=1, order=3):
        """Apply bandpass filter to extract frequency band."""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        if high >= 1:
            high = 0.99
        if low <= 0:
            low = 0.01
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        features = compute_features(data)
        
        common = si.index.intersection(features.index)
        si_clean = si.loc[common].dropna()
        adx_clean = features.loc[common, 'adx'].dropna()
        
        common_final = si_clean.index.intersection(adx_clean.index)
        si_values = si_clean.loc[common_final].values
        adx_values = adx_clean.loc[common_final].values
        
        if len(si_values) < 200:
            print("    Not enough data")
            continue
        
        # Frequency bands (in cycles per day)
        bands = {
            'very_short': (1/3, 1/7),      # 3-7 days
            'short': (1/7, 1/14),          # 7-14 days
            'medium': (1/14, 1/30),        # 14-30 days
            'long': (1/30, 1/60),          # 30-60 days
            'very_long': (1/60, 1/120),    # 60-120 days
        }
        
        band_correlations = {}
        print(f"    {'Band':<12} {'Period':>15} {'r(SI,ADX)':>12}")
        print("    " + "-"*42)
        
        for band_name, (high_freq, low_freq) in bands.items():
            try:
                si_filtered = bandpass_filter(si_values, low_freq, high_freq)
                adx_filtered = bandpass_filter(adx_values, low_freq, high_freq)
                
                r, _ = spearmanr(si_filtered, adx_filtered)
                band_correlations[band_name] = r
                
                period = f"{int(1/high_freq)}-{int(1/low_freq)} days"
                print(f"    {band_name:<12} {period:>15} {r:>+12.3f}")
            except:
                pass
        
        freq_results[name] = band_correlations
        
        # Find strongest band
        if band_correlations:
            strongest = max(band_correlations.items(), key=lambda x: abs(x[1]))
            discover(f"SI-ADX relationship strongest at {strongest[0]} frequency in {name} (r={strongest[1]:.3f})")
    
    all_results['frequency_bands'] = freq_results
    
    # ============================================================
    section("5. INDEPENDENT COMPONENT ANALYSIS")
    # ============================================================
    
    print("\n  Extracting independent components from features")
    
    ica_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        features = compute_features(data)
        
        common = si.index.intersection(features.index)
        si_aligned = si.loc[common]
        
        feature_cols = ['adx', 'rsi_extremity', 'volatility', 'momentum']
        X = features.loc[common, feature_cols].dropna()
        common_final = X.index.intersection(si_aligned.dropna().index)
        X = X.loc[common_final]
        si_final = si_aligned.loc[common_final]
        
        if len(X) < 100:
            print("    Not enough data")
            continue
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # ICA
        try:
            ica = FastICA(n_components=3, random_state=42)
            ics = ica.fit_transform(X_scaled)
            
            # Correlate SI with each IC
            ic_corrs = []
            for i in range(ics.shape[1]):
                r, _ = spearmanr(si_final, ics[:, i])
                ic_corrs.append(r)
                print(f"    SI vs IC{i+1}: r = {r:+.3f}")
            
            ica_results[name] = {
                'si_ic_correlations': ic_corrs,
            }
            
            strongest = max(range(len(ic_corrs)), key=lambda x: abs(ic_corrs[x]))
            if abs(ic_corrs[strongest]) > 0.15:
                discover(f"SI correlates with independent component IC{strongest+1} in {name} (r={ic_corrs[strongest]:.3f})")
        except:
            print("    ICA failed to converge")
    
    all_results['ica'] = ica_results
    
    # ============================================================
    section("SUMMARY")
    # ============================================================
    
    print(f"\n  Total discoveries: {len(discoveries)}")
    for i, d in enumerate(discoveries, 1):
        print(f"    {i}. {d}")
    
    out_path = Path("results/phase1_decomposition/results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'discoveries': discoveries,
            'results': all_results,
        }, f, indent=2, default=str)
    
    discoveries_path = Path("docs/DISCOVERIES_LOG.md")
    with open(discoveries_path, 'a') as f:
        f.write(f"\n\n## Phase 1 Decomposition - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        for d in discoveries:
            f.write(f"- {d}\n")
    
    print(f"\n  Results saved: {out_path}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
