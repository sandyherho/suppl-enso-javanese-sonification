#!/usr/bin/env python
"""
figure0_enso_timeseries.py

Create Beautiful ENSO Time Series Visualization
===============================================

Authors: Sandy H. S. Herho, Rusmawan Suwarman, Edi Riawan, Nurjanna J. Trilaksono, Iwan P.Anwar
Institution: Weather and Climate Prediction Laboratory (WCPL) ITB
Date: 10/11/2025
License: WTFPL
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, DateFormatter

# Configuration
INPUT_DIR = Path('../input_data')
FIGS_DIR = Path('../figs')
INPUT_FILE = INPUT_DIR / 'nino34_hadisst_mon_1870_2024.csv'
OUTPUT_FILE = FIGS_DIR / 'fig_enso_timeseries.png'


def load_enso_data(filepath):
    """Load ENSO Nino 3.4 data."""
    df = pd.read_csv(filepath, skipinitialspace=True)
    df.columns = df.columns.str.strip().str.lower()
    df['date'] = pd.to_datetime(df['date'])
    df['nino34'] = pd.to_numeric(df['nino34'], errors='coerce')
    df.loc[df['nino34'] < -900, 'nino34'] = np.nan
    df = df.dropna(subset=['nino34'])
    return df['date'].values, df['nino34'].values


def create_timeseries_plot(dates, enso_data, output_file):
    """Create beautiful ENSO time series visualization."""
    dates_dt = pd.to_datetime(dates)
    
    fig = plt.figure(figsize=(20, 8))
    ax = plt.subplot(111)
    
    # Color scheme
    el_nino_color = '#D62828'
    la_nina_color = '#003049'
    line_color = '#2B2D42'
    
    # Reference lines
    ax.axhline(y=0, color='#457B9D', linewidth=2, alpha=0.6, zorder=1)
    ax.axhline(y=0.5, color=el_nino_color, linewidth=1.2, linestyle='--', alpha=0.4, zorder=1)
    ax.axhline(y=-0.5, color=la_nina_color, linewidth=1.2, linestyle='--', alpha=0.4, zorder=1)
    ax.axhline(y=1.5, color=el_nino_color, linewidth=0.8, linestyle=':', alpha=0.3, zorder=1)
    ax.axhline(y=-1.5, color=la_nina_color, linewidth=0.8, linestyle=':', alpha=0.3, zorder=1)
    
    # Fill El Nino and La Nina regions
    ax.fill_between(dates_dt, 0, enso_data, where=(enso_data >= 0.5),
                     color=el_nino_color, alpha=0.25, label='El Niño', zorder=2)
    ax.fill_between(dates_dt, 0, enso_data, where=(enso_data <= -0.5),
                     color=la_nina_color, alpha=0.25, label='La Niña', zorder=2)
    
    # Main time series line
    ax.plot(dates_dt, enso_data, color=line_color, linewidth=1.5, alpha=0.9, zorder=3)
    
    # Highlight extreme events
    extreme_mask = np.abs(enso_data) >= 2.0
    if np.any(extreme_mask):
        ax.scatter(dates_dt[extreme_mask], enso_data[extreme_mask], 
                  s=60, c='red', alpha=0.8, zorder=4, marker='o', 
                  edgecolors='darkred', linewidths=1.5, label='Extreme Events (|T| >= 2°C)')
    
    # Axis styling
    ax.set_xlabel('Year [C.E.]', fontsize=14, fontweight='600')
    ax.set_ylabel(r'Niño 3.4 Index [$^{\circ}$C]', fontsize=14, fontweight='600')
    
    # Grid
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='upper left', framealpha=0.95, fontsize=11, 
             shadow=True, fancybox=True)
    
    # Y-axis limits with padding
    y_margin = 0.3
    ax.set_ylim(np.min(enso_data) - y_margin, np.max(enso_data) + y_margin)
    
    # X-axis formatting
    ax.xaxis.set_major_locator(YearLocator(20))
    ax.xaxis.set_minor_locator(YearLocator(5))
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    
    # Spine styling
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#333333')
    
    # Tick parameters
    ax.tick_params(axis='both', which='major', labelsize=11, length=6, width=1.5)
    ax.tick_params(axis='both', which='minor', length=3, width=1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=350, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_file.name}")


def main():
    print("\n" + "="*70)
    print("FIGURE ENSO TIME SERIES")
    print("="*70 + "\n")
    
    FIGS_DIR.mkdir(exist_ok=True)
    
    print("Loading ENSO data...")
    dates, enso_data = load_enso_data(INPUT_FILE)
    print(f"  Loaded {len(enso_data)} months of data\n")
    
    print("Creating figure...")
    create_timeseries_plot(dates, enso_data, OUTPUT_FILE)
    
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
