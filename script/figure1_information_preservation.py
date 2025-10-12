#!/usr/bin/env python
"""
figure1_information_preservation.py

Figure 1: Information Preservation Analysis
===========================================

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
OUTPUT_FILE = FIGS_DIR / 'fig1_information_preservation.png'


def load_enso_data(filepath):
    """Load ENSO Nino 3.4 data."""
    df = pd.read_csv(filepath, skipinitialspace=True)
    df.columns = df.columns.str.strip().str.lower()
    df['date'] = pd.to_datetime(df['date'])
    df['nino34'] = pd.to_numeric(df['nino34'], errors='coerce')
    df.loc[df['nino34'] < -900, 'nino34'] = np.nan
    df = df.dropna(subset=['nino34'])
    return df['nino34'].values


def safe_bins(data, default=30, min_bins=5, max_bins=100):
    """Calculate safe number of bins."""
    if len(data) < 2:
        return min_bins
    
    data_clean = data[np.isfinite(data)]
    if len(data_clean) < 2:
        return min_bins
    
    if np.std(data_clean) == 0:
        return min_bins
    
    try:
        iqr = np.percentile(data_clean, 75) - np.percentile(data_clean, 25)
        if iqr == 0:
            bins = int(np.ceil(np.log2(len(data_clean)) + 1))
        else:
            h = 2 * iqr / (len(data_clean)**(1/3))
            if h == 0:
                bins = default
            else:
                bins = int(np.ceil((np.max(data_clean) - np.min(data_clean)) / h))
    except:
        bins = default
    
    return max(min_bins, min(bins, max_bins))


def shannon_entropy(data):
    """Calculate Shannon entropy."""
    if len(data) < 2:
        return 0.0, 5
    
    data_clean = data[np.isfinite(data)]
    if len(data_clean) < 2:
        return 0.0, 5
    
    bins = safe_bins(data_clean)
    
    try:
        hist, _ = np.histogram(data_clean, bins=bins, density=True)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0, bins
        
        bin_width = (np.max(data_clean) - np.min(data_clean)) / bins
        prob = hist * bin_width
        prob = prob / np.sum(prob)
        
        entropy = -np.sum(prob * np.log2(prob + 1e-10))
        return entropy, bins
    except:
        return 0.0, bins


def sample_entropy(data, m=2, r=None):
    """Calculate Sample Entropy."""
    if r is None:
        r = 0.2 * np.std(data)
    
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


def approximate_entropy(data, m=2, r=None):
    """Calculate Approximate Entropy."""
    if r is None:
        r = 0.2 * np.std(data)
    
    N = len(data)
    if N < 10:
        return np.nan
    
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
    
    def _phi(m):
        x = [[data[j] for j in range(i, i + m)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1) 
             for x_i in x]
        return (N - m + 1)**(-1) * sum(np.log(np.array(C) + 1e-10))
    
    try:
        return abs(_phi(m) - _phi(m + 1))
    except:
        return np.nan


def create_figure1(enso_data, all_results, output_file):
    """Create Figure 1: Information Preservation Analysis."""
    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)
    
    # Panel A: ENSO Entropy Profile over time
    ax1 = fig.add_subplot(gs[0, 0])
    
    window_size = 120  # 10 years
    stride = 12  # 1 year
    
    shannon_vals = []
    sample_vals = []
    approx_vals = []
    positions = []
    
    for i in range(0, len(enso_data) - window_size, stride):
        window = enso_data[i:i+window_size]
        s_ent, _ = shannon_entropy(window)
        shannon_vals.append(s_ent)
        se = sample_entropy(window, m=2)
        sample_vals.append(se if not np.isnan(se) else 0)
        ae = approximate_entropy(window, m=2)
        approx_vals.append(ae if not np.isnan(ae) else 0)
        positions.append((i + window_size/2) / 12)  # Convert to years
    
    ax1.plot(positions, shannon_vals, label='Shannon', linewidth=2.5, color='#2E86AB', alpha=0.9)
    ax1.plot(positions, sample_vals, label='Sample', linewidth=2.5, color='#A23B72', alpha=0.9)
    ax1.plot(positions, approx_vals, label='Approximate', linewidth=2.5, color='#F18F01', alpha=0.9)
    
    ax1.set_xlabel('Time (years)', fontsize=12, fontweight='600')
    ax1.set_ylabel('Entropy', fontsize=12, fontweight='600')
    ax1.legend(loc='best', fontsize=10, framealpha=0.95)
    ax1.grid(alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.text(0.02, 0.98, 'A', transform=ax1.transAxes, fontsize=16, 
             fontweight='bold', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel B: Music Information Content
    ax2 = fig.add_subplot(gs[0, 1])
    
    modes_with_data = [r for r in all_results if r['midi_exists']]
    mode_names = [r['mode'].replace('_', '\n') for r in modes_with_data]
    pitch_entropies = [r['pitch_entropy'] for r in modes_with_data]
    velocity_entropies = [r['velocity_entropy'] for r in modes_with_data]
    
    x = np.arange(len(mode_names))
    width = 0.38
    
    ax2.bar(x - width/2, pitch_entropies, width, label='Pitch', 
           color='#06A77D', alpha=0.85, edgecolor='black', linewidth=0.8)
    ax2.bar(x + width/2, velocity_entropies, width, label='Velocity', 
           color='#D62828', alpha=0.85, edgecolor='black', linewidth=0.8)
    
    ax2.set_ylabel('Shannon Entropy (bits)', fontsize=12, fontweight='600')
    ax2.set_xticks(x)
    ax2.set_xticklabels(mode_names, fontsize=9, rotation=45, ha='right')
    ax2.legend(loc='best', fontsize=10, framealpha=0.95)
    ax2.grid(alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
    ax2.text(0.02, 0.98, 'B', transform=ax2.transAxes, fontsize=16, 
             fontweight='bold', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel C: Mutual Information Matrix
    ax3 = fig.add_subplot(gs[0, 2])
    
    mi_pitch = [r.get('mi_enso_pitch', 0) for r in modes_with_data]
    mi_velocity = [r.get('mi_enso_velocity', 0) for r in modes_with_data]
    nmi_pitch = [r.get('nmi_enso_pitch', 0) for r in modes_with_data]
    
    mi_matrix = np.array([mi_pitch, mi_velocity, nmi_pitch])
    
    im = ax3.imshow(mi_matrix, aspect='auto', cmap='YlOrRd', vmin=0, interpolation='nearest')
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(['MI(ENSO,Pitch)', 'MI(ENSO,Velocity)', 'NMI(ENSO,Pitch)'], fontsize=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(mode_names, fontsize=9, rotation=45, ha='right')
    
    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('Mutual Information (bits)', fontsize=10)
    ax3.text(0.02, 0.98, 'C', transform=ax3.transAxes, fontsize=16, 
             fontweight='bold', va='top', color='white',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
    
    # Panel D: Information Preservation
    ax4 = fig.add_subplot(gs[0, 3])
    
    enso_entropy, _ = shannon_entropy(enso_data)
    
    preservation = [(r['mi_enso_pitch'] / enso_entropy) * 100 for r in modes_with_data]
    colors_preservation = ['#2E86AB' if 'pelog' in name else '#A23B72' for name in mode_names]
    
    ax4.barh(x, preservation, color=colors_preservation, alpha=0.85, 
            edgecolor='black', linewidth=0.8)
    ax4.set_xlabel('Information Preservation (%)', fontsize=12, fontweight='600')
    ax4.set_yticks(x)
    ax4.set_yticklabels(mode_names, fontsize=9)
    ax4.grid(alpha=0.3, axis='x', linestyle='--', linewidth=0.8)
    ax4.axvline(x=50, color='red', linestyle='--', linewidth=2, alpha=0.6)
    ax4.text(0.02, 0.98, 'D', transform=ax4.transAxes, fontsize=16, 
             fontweight='bold', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.savefig(output_file, dpi=500, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_file.name}")


def main():
    print("\n" + "="*70)
    print("FIGURE 1: INFORMATION PRESERVATION")
    print("="*70 + "\n")
    
    FIGS_DIR.mkdir(exist_ok=True)
    
    print("Loading data...")
    enso_data = load_enso_data(INPUT_FILE)
    
    with open(RESULTS_FILE, 'rb') as f:
        all_results = pickle.load(f)
    
    print(f"  Loaded {len(enso_data)} months of ENSO data")
    print(f"  Loaded {len(all_results)} mode analysis results\n")
    
    print("Creating figure...")
    create_figure1(enso_data, all_results, OUTPUT_FILE)
    
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
