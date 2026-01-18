#!/usr/bin/env python3
"""Generate toy example figure showing SI emergence on synthetic data."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

def simulate_replicator_dynamics(n_replicators=10, n_niches=2, T=500, regime_length=50):
    """Simulate replicator dynamics with regime switching."""
    
    # Initialize affinities (slightly perturbed from uniform for diversity)
    np.random.seed(42)
    affinities = np.random.dirichlet(np.ones(n_niches) * 10, size=n_replicators)
    
    # Track SI over time
    si_history = []
    regime_history = []
    
    for t in range(T):
        # Determine regime
        regime = (t // regime_length) % 2
        regime_history.append(regime)
        
        # Set fitness based on regime (with some noise)
        if regime == 0:
            fitness = np.array([1.2, 0.8]) + np.random.normal(0, 0.05, 2)
        else:
            fitness = np.array([0.8, 1.2]) + np.random.normal(0, 0.05, 2)
        fitness = np.clip(fitness, 0.1, 2.0)  # Ensure positive
        
        # Update affinities via replicator dynamics
        for i in range(n_replicators):
            expected_fitness = np.dot(affinities[i], fitness)
            if expected_fitness > 0:
                affinities[i] = affinities[i] * fitness / expected_fitness
            affinities[i] = np.clip(affinities[i], 1e-6, 1)
            affinities[i] /= affinities[i].sum()  # Normalize
        
        # Compute SI
        entropies = []
        for i in range(n_replicators):
            p = affinities[i]
            p = np.clip(p, 1e-10, 1)  # Avoid log(0)
            entropy = -np.sum(p * np.log(p))
            entropies.append(entropy)
        
        mean_entropy = np.mean(entropies)
        max_entropy = np.log(n_niches)
        si = 1 - mean_entropy / max_entropy
        si_history.append(si)
    
    return np.array(si_history), np.array(regime_history), affinities

def main():
    print("Generating toy example figure...")
    
    # Simulate
    si, regimes, final_affinities = simulate_replicator_dynamics()
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    
    # Panel (a): SI evolution
    ax = axes[0, 0]
    ax.plot(si, 'b-', linewidth=1.5, label='SI(t)')
    ax.axhline(y=0.75, color='gray', linestyle='--', alpha=0.7, label='Equilibrium')
    ax.fill_between(range(len(regimes)), 0, 1, 
                    where=np.array(regimes)==0, alpha=0.2, color='green', label='Regime A')
    ax.fill_between(range(len(regimes)), 0, 1, 
                    where=np.array(regimes)==1, alpha=0.2, color='red', label='Regime B')
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('SI(t)')
    ax.set_title('(a) SI Emergence Over Time')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_ylim(0, 1)
    
    # Panel (b): Affinity evolution (for one replicator)
    ax = axes[0, 1]
    # Re-simulate to track affinity evolution
    np.random.seed(42)
    affinities_history = []
    affinities = np.random.dirichlet(np.ones(2) * 10, size=10)
    for t in range(500):
        regime = (t // 50) % 2
        fitness = np.array([1.2, 0.8]) if regime == 0 else np.array([0.8, 1.2])
        fitness = fitness + np.random.normal(0, 0.05, 2)
        fitness = np.clip(fitness, 0.1, 2.0)
        for i in range(10):
            expected_fitness = np.dot(affinities[i], fitness)
            if expected_fitness > 0:
                affinities[i] = affinities[i] * fitness / expected_fitness
            affinities[i] = np.clip(affinities[i], 1e-6, 1)
            affinities[i] /= affinities[i].sum()
        affinities_history.append(affinities[0].copy())
    
    affinities_history = np.array(affinities_history)
    ax.plot(affinities_history[:, 0], 'g-', linewidth=1.5, label='Niche 1 affinity')
    ax.plot(affinities_history[:, 1], 'r-', linewidth=1.5, label='Niche 2 affinity')
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Affinity')
    ax.set_title('(b) Replicator Affinity Dynamics')
    ax.legend(loc='center right', fontsize=8)
    ax.set_ylim(0, 1)
    
    # Panel (c): SI-regime correlation
    ax = axes[1, 0]
    # Compute rolling SI for different window sizes
    windows = [10, 30, 50, 100]
    correlations = []
    for w in windows:
        if w < len(si):
            rolling_si = np.convolve(si, np.ones(w)/w, mode='valid')
            rolling_regime = np.convolve(regimes.astype(float), np.ones(w)/w, mode='valid')
            corr = np.corrcoef(rolling_si, rolling_regime)[0, 1]
            correlations.append(corr)
    ax.bar(range(len(windows)), correlations, color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels([f'{w}' for w in windows])
    ax.set_xlabel('Rolling Window Size')
    ax.set_ylabel('SI-Regime Correlation')
    ax.set_title('(c) Correlation by Timescale')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Panel (d): Phase diagram
    ax = axes[1, 1]
    # Show SI convergence for different fitness differentials
    deltas = [0.05, 0.1, 0.2, 0.3]
    for delta in deltas:
        si_temp = []
        affinities = np.ones((10, 2)) / 2
        for t in range(200):
            fitness = np.array([1 + delta, 1 - delta])
            for i in range(10):
                expected_fitness = np.dot(affinities[i], fitness)
                affinities[i] = affinities[i] * fitness / expected_fitness
                affinities[i] /= affinities[i].sum()
            entropies = [-np.sum(affinities[j] * np.log(np.clip(affinities[j], 1e-10, 1))) 
                        for j in range(10)]
            si_temp.append(1 - np.mean(entropies) / np.log(2))
        ax.plot(si_temp, label=f'Δ = {delta}', linewidth=1.5)
    
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('SI(t)')
    ax.set_title('(d) Convergence Speed vs Fitness Differential')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save
    output_dir = Path(__file__).parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'toy_example.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'toy_example.pdf', bbox_inches='tight')
    print(f"Saved to {output_dir / 'toy_example.png'}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Final SI: {si[-1]:.3f}")
    print(f"  SI-Regime correlation: {np.corrcoef(si, regimes)[0,1]:.3f}")
    print(f"  Time to SI=0.5: {np.argmax(si > 0.5)} timesteps")

if __name__ == '__main__':
    main()
