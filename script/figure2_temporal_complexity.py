#!/usr/bin/env python
"""
figure2_temporal_complexity.py

Figure 2: Temporal Complexity Analysis
======================================

Authors: Sandy H. S. Herho, Rusmawan Suwarman, Edi Riawan, Nurjanna J. Trilaksono
Institution: Weather and Climate Prediction Laboratory (WCPL) ITB
Date: 10/11/2025
License: WTFPL
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle

# Configuration
INPUT_DIR = Path('../input_data')
STATS_DIR = Path('../stats')
FIGS_DIR = Path('../figs')

INPUT_FILE = INPUT_DIR / 'nino34_hadisst_mon_1870_2024.csv'
RESULTS_FILE = STATS_DIR / 'music_analysis_results.pkl'
OUTPUT_FILE = FIGS_DIR / 'fig2_temporal_complexity.png'


def load_enso_data(filepath):
    """Load ENSO Nino 3.4 data."""
    df = pd.read_csv(filepath, skipinitialspace=True)
    df.columns = df.columns.str.strip().str.lower()
    df['date'] = pd.to_datetime(df['date'])
    df['nino34'] = pd.to_numeric(df['nino34'], errors='coerce')
    df.loc[df['nino34'] < -900, 'nino34'] = np.nan
    df = df.dropna(subset=['nino34'])
    return df['nino34'].values


def multiscale_entropy(data, scale_max=20, m=2, r=None):
    """Calculate Multiscale Entropy."""
    if r is None:
        r = 0.2 * np.std(data)
    
    def sample_entropy(data, m, r):
        N = len(data)
        if N < 10:
            return np.nan
        
        def _maxdist(x_i, x_j):
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
        
        def _phi(m):
            x = [[data[j] for j in range(i, i + m)] for i in range(N - m + 1)]
            C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) - 1 for x_i in x]
            return sum(C) / (N - m + 1)
        
        try:
            phi_m = _phi(m)
            phi_m_plus_1 = _phi(m + 1)
            
            if phi_m == 0 or phi_m_plus_1 == 0:
                return np.nan
            
            return -np.log(phi_m_plus_1 / phi_m)
        except:
            return np.nan
    
    mse = []
    for scale in range(1, scale_max + 1):
        if len(data) < scale * 10:
            break
        coarse = [np.mean(data[i:i+scale]) for i in range(0, len(data) - scale + 1, scale)]
        if len(coarse) < 10:
            break
        se = sample_entropy(coarse, m=m, r=r)
        mse.append(se)
    
    return np.array(mse)


def create_figure2(enso_data, all_results, output_file):
    """Create Figure 2: Temporal Complexity."""
    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)
    
    modes_with_data = [r for r in all_results if r['midi_exists']]
    
    # Panel A: Multiscale Entropy
    ax1 = fig.add_subplot(gs[0, 0])
    
    print("  Calculating ENSO MSE...")
    enso_mse = multiscale_entropy(enso_data, scale_max=20)
    scales = np.arange(1, len(enso_mse) + 1)
    
    ax1.plot(scales, enso_mse, 'o-', linewidth=3.5, markersize=10, 
             label='ENSO', color='black', alpha=0.9, markeredgecolor='white', 
             markeredgewidth=1.5)
    
    ax1.set_xlabel('Scale', fontsize=12, fontweight='600')
    ax1.set_ylabel('Sample Entropy', fontsize=12, fontweight='600')
    ax1.legend(loc='best', fontsize=10, framealpha=0.95)
    ax1.grid(alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.set_xlim(0, 21)
    ax1.text(0.02, 0.98, 'A', transform=ax1.transAxes, fontsize=16, 
             fontweight='bold', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel B: Entropy Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    
    mode_names = [r['mode'].replace('_', '\n') for r in modes_with_data]
    sample_entropies = [r.get('pitch_sample_entropy', 0) for r in modes_with_data]
    approx_entropies = [r.get('pitch_approx_entropy', 0) for r in modes_with_data]
    
    # Replace NaN with 0
    sample_entropies = [0 if np.isnan(x) else x for x in sample_entropies]
    approx_entropies = [0 if np.isnan(x) else x for x in approx_entropies]
    
    x = np.arange(len(mode_names))
    width = 0.38
    
    ax2.bar(x - width/2, sample_entropies, width, label='Sample', 
           color='#A23B72', alpha=0.85, edgecolor='black', linewidth=0.8)
    ax2.bar(x + width/2, approx_entropies, width, label='Approximate', 
           color='#F18F01', alpha=0.85, edgecolor='black', linewidth=0.8)
    
    ax2.set_ylabel('Entropy', fontsize=12, fontweight='600')
    ax2.set_xticks(x)
    ax2.set_xticklabels(mode_names, fontsize=8, rotation=45, ha='right')
    ax2.legend(loc='best', fontsize=10, framealpha=0.95)
    ax2.grid(alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
    ax2.text(0.02, 0.98, 'B', transform=ax2.transAxes, fontsize=16, 
             fontweight='bold', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel C: Complexity Landscape
    ax3 = fig.add_subplot(gs[0, 2])
    
    valid_indices = [i for i, (se, ae) in enumerate(zip(sample_entropies, approx_entropies)) 
                     if se > 0 and ae > 0]
    
    if len(valid_indices) > 0:
        plot_sample = [sample_entropies[i] for i in valid_indices]
        plot_approx = [approx_entropies[i] for i in valid_indices]
        plot_labels = [mode_names[i][:12] for i in valid_indices]
        
        scatter = ax3.scatter(plot_approx, plot_sample, s=150, alpha=0.7, 
                   c=range(len(plot_sample)), cmap='tab10', edgecolors='black', linewidths=1.5)
        
        for i, label in enumerate(plot_labels):
            ax3.annotate(label, (plot_approx[i], plot_sample[i]), 
                        fontsize=7, alpha=0.8, xytext=(6, 6), 
                        textcoords='offset points', 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    ax3.set_xlabel('Approximate Entropy', fontsize=12, fontweight='600')
    ax3.set_ylabel('Sample Entropy', fontsize=12, fontweight='600')
    ax3.grid(alpha=0.3, linestyle='--', linewidth=0.8)
    ax3.text(0.02, 0.98, 'C', transform=ax3.transAxes, fontsize=16, 
             fontweight='bold', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel D: Permutation Entropy
    ax4 = fig.add_subplot(gs[0, 3])
    
    perm_entropies = [r.get('pitch_perm_entropy', 0) for r in modes_with_data]
    
    x = np.arange(len(mode_names))
    bars = ax4.bar(x, perm_entropies, color='#06A77D', alpha=0.85, 
                  edgecolor='black', linewidth=0.8)
    
    ax4.set_ylabel('Permutation Entropy', fontsize=12, fontweight='600')
    ax4.set_xticks(x)
    ax4.set_xticklabels(mode_names, fontsize=8, rotation=45, ha='right')
    ax4.grid(alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
    ax4.set_ylim(0, 1)
    ax4.text(0.02, 0.98, 'D', transform=ax4.transAxes, fontsize=16, 
             fontweight='bold', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.savefig(output_file, dpi=500, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_file.name}")


def main():
    print("\n" + "="*70)
    print("FIGURE 2: TEMPORAL COMPLEXITY")
    print("="*70 + "\n")
    
    FIGS_DIR.mkdir(exist_ok=True)
    
    print("Loading data...")
    enso_data = load_enso_data(INPUT_FILE)
    
    with open(RESULTS_FILE, 'rb') as f:
        all_results = pickle.load(f)
    
    print(f"  Loaded {len(enso_data)} months of ENSO data")
    print(f"  Loaded {len(all_results)} mode analysis results\n")
    
    print("Creating figure...")
    create_figure2(enso_data, all_results, OUTPUT_FILE)
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
