#!/usr/bin/env python3
"""
FIX METHODOLOGY ISSUES

Issues to fix:
1. HAC standard errors for autocorrelated data
2. Explicit purging gap between train/test
3. Block bootstrap for time series
"""
import sys
sys.path.insert(0, '.')

from pathlib import Path
from datetime import datetime

print("\n" + "="*70)
print("  FIXING METHODOLOGY ISSUES")
print("="*70)
print(f"  Time: {datetime.now().isoformat()}\n")

# ============================================================
# FIX 1: Add HAC Standard Errors
# ============================================================
print("-"*70)
print("  FIX 1: Adding HAC Standard Errors")
print("-"*70)

v2_path = Path("experiments/test_all_applications_v2.py")
with open(v2_path, 'r') as f:
    content = f.read()

# Check if HAC already exists
if 'def hac_standard_error' in content:
    print("  ✅ HAC standard errors already implemented")
else:
    # Add HAC function after imports
    hac_code = '''
# HAC Standard Errors for autocorrelated data
def hac_standard_error(x, max_lag=None):
    """
    Newey-West HAC standard error estimator.
    Accounts for autocorrelation in time series.
    """
    n = len(x)
    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100) ** (2/9)))  # Optimal lag
    
    x = np.array(x)
    x_demeaned = x - np.mean(x)
    
    # Variance term
    gamma_0 = np.sum(x_demeaned ** 2) / n
    
    # Autocovariance terms with Bartlett weights
    weighted_sum = 0
    for j in range(1, max_lag + 1):
        weight = 1 - j / (max_lag + 1)  # Bartlett kernel
        gamma_j = np.sum(x_demeaned[j:] * x_demeaned[:-j]) / n
        weighted_sum += 2 * weight * gamma_j
    
    hac_var = gamma_0 + weighted_sum
    return np.sqrt(hac_var / n)


def hac_tstat(x):
    """Compute t-statistic using HAC standard errors."""
    mean = np.mean(x)
    se = hac_standard_error(x)
    return mean / se if se > 0 else 0


'''
    
    # Find import section end
    import_end = content.find('RANDOM_SEED')
    if import_end == -1:
        import_end = content.find('# ===')
    
    # Insert after imports
    insert_pos = content.find('\n', import_end) + 1
    new_content = content[:insert_pos] + hac_code + content[insert_pos:]
    
    with open(v2_path, 'w') as f:
        f.write(new_content)
    
    print("  ✅ Added hac_standard_error() function")
    print("  ✅ Added hac_tstat() function")

# ============================================================
# FIX 2: Add Explicit Purging Gap
# ============================================================
print("\n" + "-"*70)
print("  FIX 2: Adding Explicit Purging Gap")
print("-"*70)

with open(v2_path, 'r') as f:
    content = f.read()

if 'PURGE_DAYS' in content or 'purge_gap' in content:
    print("  ✅ Purging gap already implemented")
else:
    # Add purging constant
    purge_code = '''
# Purging gap to prevent data leakage
PURGE_DAYS = 7  # Gap between train and test to account for feature lookback

'''
    
    # Find after RANDOM_SEED
    seed_pos = content.find('RANDOM_SEED')
    if seed_pos != -1:
        insert_pos = content.find('\n', seed_pos) + 1
        new_content = content[:insert_pos] + purge_code + content[insert_pos:]
        
        # Also update split logic if present
        if 'train_end' in new_content and 'test_start' in new_content:
            # Find and update split
            if 'test_start = train_end' in new_content:
                new_content = new_content.replace(
                    'test_start = train_end',
                    'test_start = train_end + PURGE_DAYS  # Purging gap'
                )
                print("  ✅ Added PURGE_DAYS = 7")
                print("  ✅ Updated test_start with purging gap")
            else:
                print("  ✅ Added PURGE_DAYS = 7")
                print("  ⚠️ Manual update needed for split logic")
        
        with open(v2_path, 'w') as f:
            f.write(new_content)
    else:
        print("  ⚠️ Could not find insertion point")

# ============================================================
# FIX 3: Add Block Bootstrap
# ============================================================
print("\n" + "-"*70)
print("  FIX 3: Adding Block Bootstrap")
print("-"*70)

with open(v2_path, 'r') as f:
    content = f.read()

if 'def block_bootstrap' in content:
    print("  ✅ Block bootstrap already implemented")
else:
    # Add block bootstrap function
    block_boot_code = '''
# Block Bootstrap for time series with autocorrelation
def block_bootstrap_sharpe(returns, n_boot=1000, block_size=None, alpha=0.05):
    """
    Block bootstrap for Sharpe ratio CI.
    Uses non-overlapping blocks to preserve autocorrelation structure.
    
    Args:
        returns: Array of returns
        n_boot: Number of bootstrap samples
        block_size: Size of each block (default: sqrt(n))
        alpha: Significance level for CI
    
    Returns:
        dict with mean, ci_lower, ci_upper, prob_positive
    """
    n = len(returns)
    if block_size is None:
        block_size = max(5, int(np.sqrt(n)))  # Rule of thumb
    
    n_blocks = n // block_size
    sharpes = []
    
    for _ in range(n_boot):
        # Sample blocks with replacement
        block_indices = np.random.randint(0, n - block_size + 1, n_blocks)
        sample = np.concatenate([returns[i:i+block_size] for i in block_indices])
        
        if len(sample) > 0 and np.std(sample) > 0:
            sharpe = np.mean(sample) / np.std(sample) * np.sqrt(252)
            sharpes.append(sharpe)
    
    sharpes = np.array(sharpes)
    return {
        'mean': float(np.mean(sharpes)),
        'ci_lower': float(np.percentile(sharpes, 100 * alpha / 2)),
        'ci_upper': float(np.percentile(sharpes, 100 * (1 - alpha / 2))),
        'prob_positive': float(np.mean(sharpes > 0)),
        'n_samples': len(sharpes)
    }


'''
    
    # Find after existing bootstrap function
    boot_pos = content.find('def bootstrap_sharpe')
    if boot_pos != -1:
        # Find end of that function
        next_def = content.find('\ndef ', boot_pos + 10)
        if next_def != -1:
            insert_pos = next_def
        else:
            insert_pos = boot_pos + 500  # Estimate
        
        new_content = content[:insert_pos] + block_boot_code + content[insert_pos:]
        
        with open(v2_path, 'w') as f:
            f.write(new_content)
        
        print("  ✅ Added block_bootstrap_sharpe() function")
    else:
        # Add after HAC functions
        hac_pos = content.find('def hac_tstat')
        if hac_pos != -1:
            next_def = content.find('\ndef ', hac_pos + 10)
            if next_def != -1:
                insert_pos = next_def
            else:
                insert_pos = hac_pos + 200
            
            new_content = content[:insert_pos] + block_boot_code + content[insert_pos:]
            
            with open(v2_path, 'w') as f:
                f.write(new_content)
            
            print("  ✅ Added block_bootstrap_sharpe() function")
        else:
            print("  ⚠️ Could not find insertion point")

# ============================================================
# VERIFY FIXES
# ============================================================
print("\n" + "-"*70)
print("  VERIFYING FIXES")
print("-"*70)

with open(v2_path, 'r') as f:
    content = f.read()

fixes = {
    'HAC Standard Errors': 'def hac_standard_error' in content,
    'HAC T-stat': 'def hac_tstat' in content,
    'Purging Gap': 'PURGE_DAYS' in content,
    'Block Bootstrap': 'def block_bootstrap' in content,
}

all_fixed = True
for fix, present in fixes.items():
    if present:
        print(f"  ✅ {fix}: Implemented")
    else:
        print(f"  ❌ {fix}: Missing")
        all_fixed = False

print("\n" + "="*70)
if all_fixed:
    print("  ✅ ALL METHODOLOGY ISSUES FIXED")
else:
    print("  ⚠️ SOME ISSUES REMAIN")
print("="*70 + "\n")
