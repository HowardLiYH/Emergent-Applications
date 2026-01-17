"""
Reproducibility utilities.
"""
import json
import random
import hashlib
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path
import numpy as np


def set_all_seeds(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # PyTorch if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def create_manifest(data_files: dict = None, config: dict = None) -> dict:
    """
    Create reproducibility manifest.

    Save this with your results to enable exact reproduction.
    """
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'platform': sys.platform,
        'packages': {},
        'git': {},
        'data_hashes': {},
        'config': config or {},
    }

    # Package versions
    try:
        import pkg_resources
        for pkg in ['numpy', 'pandas', 'scipy', 'statsmodels', 'scikit-learn']:
            try:
                manifest['packages'][pkg] = pkg_resources.get_distribution(pkg).version
            except Exception:
                pass
    except Exception:
        pass

    # Git info
    try:
        manifest['git']['commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode().strip()
        manifest['git']['dirty'] = len(subprocess.check_output(
            ['git', 'status', '--porcelain']
        )) > 0
    except Exception:
        manifest['git']['error'] = 'Git not available'

    # Data hashes
    if data_files:
        for name, filepath in data_files.items():
            try:
                with open(filepath, 'rb') as f:
                    manifest['data_hashes'][name] = hashlib.md5(f.read()).hexdigest()
            except Exception as e:
                manifest['data_hashes'][name] = f'error: {e}'

    return manifest


def save_manifest(manifest: dict, filepath: str = "results/manifest.json"):
    """Save manifest to JSON."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"✅ Manifest saved: {filepath}")
