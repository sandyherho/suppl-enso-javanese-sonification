#!/usr/bin/env python
"""
music_state_ts_analysis.py

Time Series Visualization and Analysis of Musical State Variables
=================================================================

Creates time series plots and statistical analysis for:
- Spectral Centroid (Brightness)
- RMS Energy (Intensity)

Outputs:
1. Four 2x2 figures:
   - fig_timeseries_brightness_pelog.png
   - fig_timeseries_brightness_slendro.png
   - fig_timeseries_energy_pelog.png
   - fig_timeseries_energy_slendro.png

2. Statistical report:
   - stats/timeseries_statistics.txt

Authors: Sandy H. S. Herho, Rusmawan Suwarman, Nurjanna J. Trilaksono
Institution: Weather and Climate Prediction Laboratory (WCPL) ITB
Date: 10/12/2025
License: WTFPL
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from datetime import datetime

# Configuration
OUTPUT_DIR = Path('../output_data')
FIGS_DIR = Path('../figs')
STATS_DIR = Path('../stats')

# Mode groups
PELOG_MODES = ['pelog_layered', 'pelog_alternating', 'pelog_melodic', 'pelog_spectral']
SLENDRO_MODES = ['slendro_layered', 'slendro_alternating', 'slendro_melodic', 'slendro_spectral']

# Colors for each composition type
COLORS = {
    'layered': '#E63946',
    'alternating': '#F77F00',
    'melodic': '#06A77D',
    'spectral': '#1D3557'
}


# ============================================================================
# TIME SERIES STATISTICS
# ============================================================================

def calculate_timeseries_stats(data, name):
    """Calculate comprehensive statistics for a time series."""
    
    stats = {
        'name': name,
        'n_points': len(data),
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'range': np.max(data) - np.min(data),
        'median': np.median(data),
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75),
        'iqr': np.percentile(data, 75) - np.percentile(data, 25),
        'skewness': scipy_stats.skew(data),
        'kurtosis': scipy_stats.kurtosis(data),
        'cv': np.std(data) / np.mean(data) if np.mean(data) != 0 else 0
    }
    
    # Trend analysis (linear regression)
    x = np.arange(len(data))
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, data)
    
    stats['trend_slope'] = slope
    stats['trend_r2'] = r_value**2
    stats['trend_pvalue'] = p_value
    
    # Rate of change statistics
    if len(data) > 1:
        rate_of_change = np.diff(data)
        stats['roc_mean'] = np.mean(rate_of_change)
        stats['roc_std'] = np.std(rate_of_change)
        stats['roc_max'] = np.max(rate_of_change)
        stats['roc_min'] = np.min(rate_of_change)
    else:
        stats['roc_mean'] = 0
        stats['roc_std'] = 0
        stats['roc_max'] = 0
        stats['roc_min'] = 0
    
    # Autocorrelation at lag 1
    if len(data) > 2:
        autocorr = np.corrcoef(data[:-1], data[1:])[0, 1]
        stats['autocorr_lag1'] = autocorr
    else:
        stats['autocorr_lag1'] = 0
    
    return stats


# ============================================================================
# TIME SERIES PLOTTING
# ============================================================================

def create_timeseries_plot(modes, variable, scale_name, output_file):
    """
    Create 2x2 time series plot.
    
    Args:
        modes: List of 4 mode names
        variable: 'brightness' or 'energy'
        scale_name: 'Pelog' or 'Slendro'
        output_file: Path to save figure
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']
    
    # Determine which column to plot and units
    if variable == 'brightness':
        data_col = 'spectral_centroid_smooth'
        ylabel = 'Brightness [Hz]'
    else:  # energy
        data_col = 'rms_energy_smooth'
        ylabel = 'Energy [amplitude]'
    
    for idx, mode in enumerate(modes):
        ax = axes[idx]
        
        # Load data
        data_file = OUTPUT_DIR / f'state_variables_{mode}.csv'
        
        if not data_file.exists():
            ax.text(0.5, 0.5, 'Data not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(ylabel)
            continue
        
        df = pd.read_csv(data_file)
        
        time = df['time_sec'].values
        data = df[data_col].values
        
        # Determine composition type for color
        comp_type = mode.split('_')[1]
        color = COLORS[comp_type]
        
        # Plot time series
        ax.plot(time, data, color=color, linewidth=1.2, alpha=0.8)
        
        # Subplot label
        ax.text(0.02, 0.98, subplot_labels[idx], transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Axes labels
        ax.set_xlabel('Time [s]', fontsize=11, fontweight='600')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='600')
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Tick parameters
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=350, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved: {output_file.name}")


# ============================================================================
# STATISTICAL REPORT
# ============================================================================

def write_timeseries_report(all_stats, output_file):
    """Write comprehensive time series statistics report."""
    
    with open(output_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("TIME SERIES STATISTICS OF MUSICAL STATE VARIABLES\n")
        f.write("="*100 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Authors: Sandy H. S. Herho, Rusmawan Suwarman, Edi Riawan, Nurjanna J. Trilaksono\n")
        f.write("Institution: Weather and Climate Prediction Laboratory (WCPL) ITB\n")
        f.write("="*100 + "\n\n")
        
        f.write("STATE VARIABLES\n")
        f.write("-"*100 + "\n")
        f.write("1. Spectral Centroid (Brightness): Center of mass of audio spectrum [Hz]\n")
        f.write("2. RMS Energy (Intensity): Root-mean-square amplitude [arbitrary units]\n\n")
        
        f.write("STATISTICS COMPUTED\n")
        f.write("-"*100 + "\n")
        f.write("Central Tendency: mean, median, quartiles\n")
        f.write("Dispersion: standard deviation, range, IQR, coefficient of variation\n")
        f.write("Distribution Shape: skewness, kurtosis\n")
        f.write("Temporal Dynamics: trend (linear regression), rate of change, autocorrelation\n")
        f.write("\n\n")
        
        # Separate stats by variable
        brightness_stats = [s for s in all_stats if 'brightness' in s['name']]
        energy_stats = [s for s in all_stats if 'energy' in s['name']]
        
        # ===== BRIGHTNESS STATISTICS =====
        f.write("="*100 + "\n")
        f.write("SPECTRAL CENTROID (BRIGHTNESS) STATISTICS\n")
        f.write("="*100 + "\n\n")
        
        # Summary table
        f.write("SUMMARY TABLE\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Mode':<25} {'Mean [Hz]':<12} {'Std [Hz]':<12} {'Range [Hz]':<12} {'Trend':<10}\n")
        f.write("-"*100 + "\n")
        
        for s in brightness_stats:
            mode_name = s['name'].replace('_brightness', '')
            trend_dir = "↑" if s['trend_slope'] > 0 else "↓"
            f.write(f"{mode_name:<25} {s['mean']:>10.2f}  {s['std']:>10.2f}  ")
            f.write(f"{s['range']:>10.2f}  {trend_dir:>8}\n")
        
        f.write("\n\n")
        
        # Detailed statistics
        f.write("DETAILED STATISTICS\n")
        f.write("-"*100 + "\n\n")
        
        for s in brightness_stats:
            mode_name = s['name'].replace('_brightness', '')
            f.write(f"MODE: {mode_name.upper()}\n")
            f.write("-"*100 + "\n")
            
            f.write(f"Central Tendency:\n")
            f.write(f"  Mean:              {s['mean']:>12.2f} Hz\n")
            f.write(f"  Median:            {s['median']:>12.2f} Hz\n")
            f.write(f"  25th percentile:   {s['q25']:>12.2f} Hz\n")
            f.write(f"  75th percentile:   {s['q75']:>12.2f} Hz\n\n")
            
            f.write(f"Dispersion:\n")
            f.write(f"  Std Deviation:     {s['std']:>12.2f} Hz\n")
            f.write(f"  Min:               {s['min']:>12.2f} Hz\n")
            f.write(f"  Max:               {s['max']:>12.2f} Hz\n")
            f.write(f"  Range:             {s['range']:>12.2f} Hz\n")
            f.write(f"  IQR:               {s['iqr']:>12.2f} Hz\n")
            f.write(f"  Coeff. Variation:  {s['cv']:>12.4f}\n\n")
            
            f.write(f"Distribution Shape:\n")
            f.write(f"  Skewness:          {s['skewness']:>12.4f}\n")
            f.write(f"  Kurtosis:          {s['kurtosis']:>12.4f}\n\n")
            
            f.write(f"Temporal Dynamics:\n")
            f.write(f"  Trend slope:       {s['trend_slope']:>12.6f} Hz/frame\n")
            f.write(f"  Trend R²:          {s['trend_r2']:>12.4f}\n")
            f.write(f"  Trend p-value:     {s['trend_pvalue']:>12.6f}\n")
            f.write(f"  Autocorr (lag-1):  {s['autocorr_lag1']:>12.4f}\n")
            f.write(f"  ROC mean:          {s['roc_mean']:>12.4f} Hz/frame\n")
            f.write(f"  ROC std:           {s['roc_std']:>12.4f} Hz/frame\n\n")
            
            # Interpretation
            f.write(f"Interpretation:\n")
            
            if abs(s['skewness']) < 0.5:
                f.write(f"  Distribution:  Nearly symmetric\n")
            elif s['skewness'] > 0:
                f.write(f"  Distribution:  Right-skewed (tail toward higher frequencies)\n")
            else:
                f.write(f"  Distribution:  Left-skewed (tail toward lower frequencies)\n")
            
            if s['trend_pvalue'] < 0.05:
                direction = "increasing" if s['trend_slope'] > 0 else "decreasing"
                f.write(f"  Trend:         Significant {direction} trend (p < 0.05)\n")
            else:
                f.write(f"  Trend:         No significant trend\n")
            
            if s['autocorr_lag1'] > 0.7:
                f.write(f"  Persistence:   High (values change slowly)\n")
            elif s['autocorr_lag1'] > 0.3:
                f.write(f"  Persistence:   Moderate\n")
            else:
                f.write(f"  Persistence:   Low (values change rapidly)\n")
            
            f.write("\n")
        
        # ===== ENERGY STATISTICS =====
        f.write("\n" + "="*100 + "\n")
        f.write("RMS ENERGY (INTENSITY) STATISTICS\n")
        f.write("="*100 + "\n\n")
        
        # Summary table
        f.write("SUMMARY TABLE\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Mode':<25} {'Mean':<12} {'Std':<12} {'Range':<12} {'Trend':<10}\n")
        f.write("-"*100 + "\n")
        
        for s in energy_stats:
            mode_name = s['name'].replace('_energy', '')
            trend_dir = "↑" if s['trend_slope'] > 0 else "↓"
            f.write(f"{mode_name:<25} {s['mean']:>10.6f}  {s['std']:>10.6f}  ")
            f.write(f"{s['range']:>10.6f}  {trend_dir:>8}\n")
        
        f.write("\n\n")
        
        # Detailed statistics
        f.write("DETAILED STATISTICS\n")
        f.write("-"*100 + "\n\n")
        
        for s in energy_stats:
            mode_name = s['name'].replace('_energy', '')
            f.write(f"MODE: {mode_name.upper()}\n")
            f.write("-"*100 + "\n")
            
            f.write(f"Central Tendency:\n")
            f.write(f"  Mean:              {s['mean']:>12.6f}\n")
            f.write(f"  Median:            {s['median']:>12.6f}\n")
            f.write(f"  25th percentile:   {s['q25']:>12.6f}\n")
            f.write(f"  75th percentile:   {s['q75']:>12.6f}\n\n")
            
            f.write(f"Dispersion:\n")
            f.write(f"  Std Deviation:     {s['std']:>12.6f}\n")
            f.write(f"  Min:               {s['min']:>12.6f}\n")
            f.write(f"  Max:               {s['max']:>12.6f}\n")
            f.write(f"  Range:             {s['range']:>12.6f}\n")
            f.write(f"  IQR:               {s['iqr']:>12.6f}\n")
            f.write(f"  Coeff. Variation:  {s['cv']:>12.4f}\n\n")
            
            f.write(f"Distribution Shape:\n")
            f.write(f"  Skewness:          {s['skewness']:>12.4f}\n")
            f.write(f"  Kurtosis:          {s['kurtosis']:>12.4f}\n\n")
            
            f.write(f"Temporal Dynamics:\n")
            f.write(f"  Trend slope:       {s['trend_slope']:>12.9f} /frame\n")
            f.write(f"  Trend R²:          {s['trend_r2']:>12.4f}\n")
            f.write(f"  Trend p-value:     {s['trend_pvalue']:>12.6f}\n")
            f.write(f"  Autocorr (lag-1):  {s['autocorr_lag1']:>12.4f}\n")
            f.write(f"  ROC mean:          {s['roc_mean']:>12.6f} /frame\n")
            f.write(f"  ROC std:           {s['roc_std']:>12.6f} /frame\n\n")
            
            # Interpretation
            f.write(f"Interpretation:\n")
            
            if abs(s['skewness']) < 0.5:
                f.write(f"  Distribution:  Nearly symmetric\n")
            elif s['skewness'] > 0:
                f.write(f"  Distribution:  Right-skewed (tail toward higher energy)\n")
            else:
                f.write(f"  Distribution:  Left-skewed (tail toward lower energy)\n")
            
            if s['trend_pvalue'] < 0.05:
                direction = "increasing" if s['trend_slope'] > 0 else "decreasing"
                f.write(f"  Trend:         Significant {direction} trend (p < 0.05)\n")
            else:
                f.write(f"  Trend:         No significant trend\n")
            
            if s['autocorr_lag1'] > 0.7:
                f.write(f"  Persistence:   High (energy changes slowly)\n")
            elif s['autocorr_lag1'] > 0.3:
                f.write(f"  Persistence:   Moderate\n")
            else:
                f.write(f"  Persistence:   Low (energy changes rapidly)\n")
            
            f.write("\n")
        
        # Comparative analysis
        f.write("\n" + "="*100 + "\n")
        f.write("COMPARATIVE ANALYSIS\n")
        f.write("="*100 + "\n\n")
        
        # By scale family
        pelog_bright = [s for s in brightness_stats if 'pelog' in s['name']]
        slendro_bright = [s for s in brightness_stats if 'slendro' in s['name']]
        
        pelog_energy = [s for s in energy_stats if 'pelog' in s['name']]
        slendro_energy = [s for s in energy_stats if 'slendro' in s['name']]
        
        f.write("BY SCALE FAMILY\n")
        f.write("-"*100 + "\n\n")
        
        f.write(f"Pelog Brightness (n={len(pelog_bright)}):\n")
        f.write(f"  Average Mean:      {np.mean([s['mean'] for s in pelog_bright]):>12.2f} Hz\n")
        f.write(f"  Average Std:       {np.mean([s['std'] for s in pelog_bright]):>12.2f} Hz\n")
        f.write(f"  Average CV:        {np.mean([s['cv'] for s in pelog_bright]):>12.4f}\n\n")
        
        f.write(f"Slendro Brightness (n={len(slendro_bright)}):\n")
        f.write(f"  Average Mean:      {np.mean([s['mean'] for s in slendro_bright]):>12.2f} Hz\n")
        f.write(f"  Average Std:       {np.mean([s['std'] for s in slendro_bright]):>12.2f} Hz\n")
        f.write(f"  Average CV:        {np.mean([s['cv'] for s in slendro_bright]):>12.4f}\n\n")
        
        f.write(f"Pelog Energy (n={len(pelog_energy)}):\n")
        f.write(f"  Average Mean:      {np.mean([s['mean'] for s in pelog_energy]):>12.6f}\n")
        f.write(f"  Average Std:       {np.mean([s['std'] for s in pelog_energy]):>12.6f}\n")
        f.write(f"  Average CV:        {np.mean([s['cv'] for s in pelog_energy]):>12.4f}\n\n")
        
        f.write(f"Slendro Energy (n={len(slendro_energy)}):\n")
        f.write(f"  Average Mean:      {np.mean([s['mean'] for s in slendro_energy]):>12.6f}\n")
        f.write(f"  Average Std:       {np.mean([s['std'] for s in slendro_energy]):>12.6f}\n")
        f.write(f"  Average CV:        {np.mean([s['cv'] for s in slendro_energy]):>12.4f}\n\n")
        
        f.write("="*100 + "\n")
        f.write("END OF TIME SERIES STATISTICS\n")
        f.write("="*100 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("TIME SERIES VISUALIZATION & ANALYSIS")
    print("="*70 + "\n")
    
    FIGS_DIR.mkdir(exist_ok=True)
    STATS_DIR.mkdir(exist_ok=True)
    
    # Check for data files
    print("Checking for state variable files...")
    all_modes = PELOG_MODES + SLENDRO_MODES
    available_modes = [m for m in all_modes if (OUTPUT_DIR / f'state_variables_{m}.csv').exists()]
    
    if len(available_modes) == 0:
        print("\nERROR: No state variable files found!")
        print(f"Expected location: {OUTPUT_DIR}/")
        print("Please run music_characterization_revised.py first.")
        return
    
    print(f"  Found {len(available_modes)}/{len(all_modes)} mode files\n")
    
    # Calculate statistics for all modes
    print("Calculating time series statistics...")
    all_stats = []
    
    for mode in available_modes:
        data_file = OUTPUT_DIR / f'state_variables_{mode}.csv'
        df = pd.read_csv(data_file)
        
        # Brightness statistics
        brightness_data = df['spectral_centroid_smooth'].values
        brightness_stats = calculate_timeseries_stats(brightness_data, f'{mode}_brightness')
        all_stats.append(brightness_stats)
        
        # Energy statistics
        energy_data = df['rms_energy_smooth'].values
        energy_stats = calculate_timeseries_stats(energy_data, f'{mode}_energy')
        all_stats.append(energy_stats)
        
        print(f"  ✓ {mode}")
    
    # Create time series plots
    print("\nCreating time series plots...")
    
    # Brightness plots
    pelog_available = [m for m in PELOG_MODES if m in available_modes]
    slendro_available = [m for m in SLENDRO_MODES if m in available_modes]
    
    if len(pelog_available) == 4:
        output = FIGS_DIR / 'fig_timeseries_brightness_pelog.png'
        create_timeseries_plot(pelog_available, 'brightness', 'Pelog', output)
    
    if len(slendro_available) == 4:
        output = FIGS_DIR / 'fig_timeseries_brightness_slendro.png'
        create_timeseries_plot(slendro_available, 'brightness', 'Slendro', output)
    
    # Energy plots
    if len(pelog_available) == 4:
        output = FIGS_DIR / 'fig_timeseries_energy_pelog.png'
        create_timeseries_plot(pelog_available, 'energy', 'Pelog', output)
    
    if len(slendro_available) == 4:
        output = FIGS_DIR / 'fig_timeseries_energy_slendro.png'
        create_timeseries_plot(slendro_available, 'energy', 'Slendro', output)
    
    # Write statistical report
    print("\nWriting statistical report...")
    report_file = STATS_DIR / 'timeseries_statistics.txt'
    write_timeseries_report(all_stats, report_file)
    print(f"  ✓ {report_file.name}")
    
    # Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    print(f"\nFigures saved to:  {FIGS_DIR}/")
    print(f"  - fig_timeseries_brightness_pelog.png")
    print(f"  - fig_timeseries_brightness_slendro.png")
    print(f"  - fig_timeseries_energy_pelog.png")
    print(f"  - fig_timeseries_energy_slendro.png")
    
    print(f"\nStatistics saved to: {STATS_DIR}/timeseries_statistics.txt")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
