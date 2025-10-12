#!/usr/bin/env python
"""
figure3_spectral_analysis.py

Figure 3: Spectral Analysis
============================

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
from scipy import signal
import pickle

# Configuration
INPUT_DIR = Path('../input_data')
STATS_DIR = Path('../stats')
FIGS_DIR = Path('../figs')

INPUT_FILE = INPUT_DIR / 'nino34_hadisst_mon_1870_2024.csv'
RESULTS_FILE = STATS_DIR / 'music_analysis_results.pkl'
OUTPUT_FILE = FIGS_DIR / 'fig3_spectral_analysis.png'


def load_enso_data(filepath):
    """Load ENSO Nino 3.4 data."""
    df = pd.read_csv(filepath, skipinitialspace=True)
    df.columns = df.columns.str.strip().str.lower()
    df['date'] = pd.to_datetime(df['date'])
    df['nino34'] = pd.to_numeric(df['nino34'], errors='coerce')
    df.loc[df['nino34'] < -900, 'nino34'] = np.nan
    df = df.dropna(subset=['nino34'])
    return df['nino34'].values


def create_figure3(enso_data, all_results, output_file):
    """Create Figure 3: Spectral Analysis."""
    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)
    
    modes_with_data = [r for r in all_results if r['midi_exists']]
    
    # Panel A: PSD Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    
    print("  Calculating power spectra...")
    fs = 12  # Monthly data = 12 samples/year
    freqs, psd_enso = signal.welch(enso_data, fs=fs, nperseg=min(512, len(enso_data)//4))
    periods = 1 / (freqs + 1e-10) / 12  # Convert to years
    
    ax1.semilogy(periods, psd_enso, linewidth=3.5, color='black', 
                label='ENSO', alpha=0.9)
    
    ax1.set_xlabel('Period (years)', fontsize=12, fontweight='600')
    ax1.set_ylabel('Power Spectral Density', fontsize=12, fontweight='600')
    ax1.legend(loc='best', fontsize=10, framealpha=0.95)
    ax1.grid(alpha=0.3, which='both', linestyle='--', linewidth=0.8)
    ax1.set_xlim(0, 20)
    ax1.text(0.02, 0.98, 'A', transform=ax1.transAxes, fontsize=16, 
             fontweight='bold', va='top', color='white',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Panel B: Spectral Entropy
    ax2 = fig.add_subplot(gs[0, 1])
    
    spec_entropies = [r.get('pitch_spectral_entropy', 0) for r in modes_with_data]
    mode_names = [r['mode'].replace('_', '\n') for r in modes_with_data]
    
    x = np.arange(len(mode_names))
    bars = ax2.bar(x, spec_entropies, color='#F18F01', alpha=0.85, 
                  edgecolor='black', linewidth=0.8)
    
    ax2.set_ylabel('Spectral Entropy', fontsize=12, fontweight='600')
    ax2.set_xticks(x)
    ax2.set_xticklabels(mode_names, fontsize=8, rotation=45, ha='right')
    ax2.grid(alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
    ax2.set_ylim(0, 1)
    ax2.text(0.02, 0.98, 'B', transform=ax2.transAxes, fontsize=16, 
             fontweight='bold', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel C: Information Metrics Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    
    mi_values = [r.get('mi_enso_pitch', 0) for r in modes_with_data]
    nmi_values = [r.get('nmi_enso_pitch', 0) for r in modes_with_data]
    
    x = np.arange(len(mode_names))
    width = 0.38
    
    ax3.bar(x - width/2, mi_values, width, label='MI', 
           color='#2E86AB', alpha=0.85, edgecolor='black', linewidth=0.8)
    ax3.bar(x + width/2, nmi_values, width, label='NMI', 
           color='#A23B72', alpha=0.85, edgecolor='black', linewidth=0.8)
    
    ax3.set_ylabel('Information Metric', fontsize=12, fontweight='600')
    ax3.set_xticks(x)
    ax3.set_xticklabels(mode_names, fontsize=8, rotation=45, ha='right')
    ax3.legend(loc='best', fontsize=10, framealpha=0.95)
    ax3.grid(alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
    ax3.text(0.02, 0.98, 'C', transform=ax3.transAxes, fontsize=16, 
             fontweight='bold', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel D: Summary Heatmap
    ax4 = fig.add_subplot(gs[0, 3])
    
    metrics = ['MI', 'NMI', 'SE', 'ApEn', 'PE']
    summary_matrix = []
    
    for r in modes_with_data:
        row = []
        row.append(r.get('mi_enso_pitch', 0))
        row.append(r.get('nmi_enso_pitch', 0))
        se = r.get('pitch_sample_entropy', 0)
        row.append(se if not np.isnan(se) else 0)
        ae = r.get('pitch_approx_entropy', 0)
        row.append(ae if not np.isnan(ae) else 0)
        row.append(r.get('pitch_perm_entropy', 0))
        summary_matrix.append(row)
    
    summary_matrix = np.array(summary_matrix).T
    
    # Normalize each row
    for i in range(summary_matrix.shape[0]):
        row_max = np.max(summary_matrix[i])
        if row_max > 0:
            summary_matrix[i] = summary_matrix[i] / row_max
    
    im = ax4.imshow(summary_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                   interpolation='nearest')
    
    ax4.set_yticks(np.arange(len(metrics)))
    ax4.set_yticklabels(metrics, fontsize=11, fontweight='600')
    ax4.set_xticks(np.arange(len(mode_names)))
    ax4.set_xticklabels(mode_names, fontsize=8, rotation=45, ha='right')
    
    cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Score', fontsize=10)
    ax4.text(0.02, 0.98, 'D', transform=ax4.transAxes, fontsize=16, 
             fontweight='bold', va='top', color='white',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
    
    plt.savefig(output_file, dpi=500, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_file.name}")


def main():
    print("\n" + "="*70)
    print("FIGURE 3: SPECTRAL ANALYSIS")
    print("="*70 + "\n")
    
    FIGS_DIR.mkdir(exist_ok=True)
    
    print("Loading data...")
    enso_data = load_enso_data(INPUT_FILE)
    
    with open(RESULTS_FILE, 'rb') as f:
        all_results = pickle.load(f)
    
    print(f"  Loaded {len(enso_data)} months of ENSO data")
    print(f"  Loaded {len(all_results)} mode analysis results\n")
    
    print("Creating figure...")
    create_figure3(enso_data, all_results, OUTPUT_FILE)
    
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
