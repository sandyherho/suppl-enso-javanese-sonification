#!/usr/bin/env python
"""
phase_space_analysis.py

Phase Space Analysis of Musical State Variables
===============================================

Creates phase space plots (Brightness vs Energy) for all sonification modes
and performs quantitative phase space analysis.

Outputs:
1. Two figures: fig_phase_space_pelog.png, fig_phase_space_slendro.png
2. Phase space analysis report: stats/phase_space_analysis.txt
3. Phase space metrics CSV: stats/phase_space_metrics.csv

Authors: Sandy H. S. Herho, Rusmawan Suwarman, Nurjanna J. Trilaksono, Iwan P. Anwar, Faiz R. Fajary
Institution: Weather and Climate Prediction Laboratory (WCPL) ITB
Date: 10/12/2025
License: WTFPL
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.stats import pearsonr
import csv
from datetime import datetime

# Configuration
OUTPUT_DIR = Path('../output_data')
FIGS_DIR = Path('../figs')
STATS_DIR = Path('../stats')

# Mode groups
PELOG_MODES = ['pelog_layered', 'pelog_alternating', 'pelog_melodic', 'pelog_spectral']
SLENDRO_MODES = ['slendro_layered', 'slendro_alternating', 'slendro_melodic', 'slendro_spectral']

# Plot configuration
COLORS = {
    'layered': '#E63946',      # Red
    'alternating': '#F77F00',  # Orange
    'melodic': '#06A77D',      # Green
    'spectral': '#1D3557'      # Navy
}

LABELS = {
    'layered': 'Layered',
    'alternating': 'Alternating',
    'melodic': 'Melodic',
    'spectral': 'Spectral'
}


# ============================================================================
# PHASE SPACE METRICS
# ============================================================================

def calculate_phase_space_area(x, y):
    """Calculate area of phase space trajectory using convex hull."""
    if len(x) < 3:
        return 0
    
    points = np.column_stack([x, y])
    
    # Remove duplicates
    points = np.unique(points, axis=0)
    
    if len(points) < 3:
        return 0
    
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume = area
    except:
        return 0


def calculate_trajectory_length(x, y):
    """Calculate total path length of trajectory."""
    if len(x) < 2:
        return 0
    
    dx = np.diff(x)
    dy = np.diff(y)
    distances = np.sqrt(dx**2 + dy**2)
    
    return np.sum(distances)


def calculate_centroid(x, y):
    """Calculate centroid of trajectory."""
    return np.mean(x), np.mean(y)


def calculate_trajectory_spread(x, y):
    """Calculate spread (standard deviation) of trajectory."""
    return np.std(x), np.std(y)


def calculate_trajectory_correlation(x, y):
    """Calculate correlation between brightness and energy."""
    if len(x) < 3:
        return 0, 1
    
    try:
        corr, pval = pearsonr(x, y)
        return corr, pval
    except:
        return 0, 1


def calculate_revisit_rate(x, y, threshold=0.05):
    """
    Calculate how often trajectory revisits similar regions.
    Higher = more circular/recurring patterns.
    """
    if len(x) < 10:
        return 0
    
    # Sample every 10th point to reduce computation
    x_sample = x[::10]
    y_sample = y[::10]
    
    revisits = 0
    total = 0
    
    for i in range(len(x_sample)):
        for j in range(i + 5, len(x_sample)):  # Skip nearby points
            dist = np.sqrt((x_sample[i] - x_sample[j])**2 + 
                          (y_sample[i] - y_sample[j])**2)
            
            if dist < threshold:
                revisits += 1
            
            total += 1
    
    return revisits / total if total > 0 else 0


def calculate_exploration_index(x, y):
    """
    Exploration index: ratio of area to path length.
    High = efficient exploration, Low = meandering.
    """
    area = calculate_phase_space_area(x, y)
    length = calculate_trajectory_length(x, y)
    
    if length > 0:
        return area / length
    return 0


def analyze_phase_space(mode_name, df):
    """Calculate all phase space metrics for a mode."""
    x = df['spectral_centroid_smooth_norm'].values
    y = df['rms_energy_smooth_norm'].values
    
    area = calculate_phase_space_area(x, y)
    length = calculate_trajectory_length(x, y)
    centroid_x, centroid_y = calculate_centroid(x, y)
    spread_x, spread_y = calculate_trajectory_spread(x, y)
    corr, pval = calculate_trajectory_correlation(x, y)
    revisit = calculate_revisit_rate(x, y)
    exploration = calculate_exploration_index(x, y)
    
    return {
        'mode': mode_name,
        'area': area,
        'path_length': length,
        'centroid_brightness': centroid_x,
        'centroid_energy': centroid_y,
        'spread_brightness': spread_x,
        'spread_energy': spread_y,
        'correlation': corr,
        'correlation_pvalue': pval,
        'revisit_rate': revisit,
        'exploration_index': exploration
    }


# ============================================================================
# PLOTTING
# ============================================================================

def create_phase_space_plot(modes, scale_name, output_file):
    """
    Create 2x2 phase space plot for a scale family.
    Simple scatter plots with consistent single color per subplot.
    
    Args:
        modes: List of mode names (4 modes)
        scale_name: 'Pelog' or 'Slendro'
        output_file: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']
    
    for idx, mode in enumerate(modes):
        ax = axes[idx]
        
        # Load data
        data_file = OUTPUT_DIR / f'state_variables_{mode}.csv'
        
        if not data_file.exists():
            ax.text(0.5, 0.5, 'Data not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel('Brightness [normalized]')
            ax.set_ylabel('Energy [normalized]')
            continue
        
        df = pd.read_csv(data_file)
        
        x = df['spectral_centroid_smooth_norm'].values
        y = df['rms_energy_smooth_norm'].values
        
        # Determine composition type for color
        comp_type = mode.split('_')[1]  # layered, alternating, melodic, spectral
        color = COLORS[comp_type]
        
        # Simple scatter plot with single consistent color
        ax.scatter(x, y, c=color, s=3, alpha=0.5, edgecolors='none')
        
        # Subplot label only
        ax.text(0.02, 0.98, subplot_labels[idx], transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Axes labels with units
        ax.set_xlabel('Brightness [normalized]', fontsize=11, fontweight='600')
        ax.set_ylabel('Energy [normalized]', fontsize=11, fontweight='600')
        
        # Set equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Set limits
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        # Tick parameters
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=350, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved: {output_file.name}")


# ============================================================================
# ANALYSIS REPORT
# ============================================================================

def write_phase_space_report(all_metrics, output_file):
    """Write detailed phase space analysis report."""
    
    with open(output_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("PHASE SPACE ANALYSIS OF MUSICAL STATE VARIABLES\n")
        f.write("="*100 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Authors: Sandy H. S. Herho, Rusmawan Suwarman, Edi Riawan, Nurjanna J. Trilaksono\n")
        f.write("Institution: Weather and Climate Prediction Laboratory (WCPL) ITB\n")
        f.write("="*100 + "\n\n")
        
        f.write("PHASE SPACE DEFINITION\n")
        f.write("-"*100 + "\n")
        f.write("Phase space: 2D space with normalized axes (0-1 scale)\n")
        f.write("  X-axis: Brightness (spectral centroid normalized) [dimensionless]\n")
        f.write("  Y-axis: Energy (RMS amplitude normalized) [dimensionless]\n")
        f.write("Trajectory: Path traced by music through this space over time\n\n")
        
        f.write("METRICS COMPUTED\n")
        f.write("-"*100 + "\n")
        f.write("  - Area: Convex hull area covered by trajectory [normalized²]\n")
        f.write("  - Path Length: Total distance traveled in phase space [normalized]\n")
        f.write("  - Centroid: Average position in phase space [normalized]\n")
        f.write("  - Spread: Standard deviation in each dimension [normalized]\n")
        f.write("  - Correlation: Linear relationship between brightness and energy [dimensionless]\n")
        f.write("  - Revisit Rate: Fraction of trajectory returning to similar regions [dimensionless]\n")
        f.write("  - Exploration Index: Efficiency of space exploration (area/length) [normalized]\n")
        f.write("\n\n")
        
        # Summary table
        f.write("SUMMARY TABLE\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Mode':<25} {'Area':<10} {'Length':<10} {'Revisit':<10} {'Explor':<10} {'Corr':<10}\n")
        f.write(f"{'':25} {'[norm²]':<10} {'[norm]':<10} {'[-]':<10} {'[norm]':<10} {'[-]':<10}\n")
        f.write("-"*100 + "\n")
        
        for m in all_metrics:
            f.write(f"{m['mode']:<25} ")
            f.write(f"{m['area']:>8.4f}  ")
            f.write(f"{m['path_length']:>8.3f}  ")
            f.write(f"{m['revisit_rate']:>8.3f}  ")
            f.write(f"{m['exploration_index']:>8.4f}  ")
            f.write(f"{m['correlation']:>8.3f}\n")
        
        f.write("\n\n")
        
        # Comparison by scale family
        f.write("COMPARISON BY SCALE FAMILY\n")
        f.write("-"*100 + "\n\n")
        
        pelog = [m for m in all_metrics if 'pelog' in m['mode']]
        slendro = [m for m in all_metrics if 'slendro' in m['mode']]
        
        f.write(f"Pelog (n={len(pelog)}):\n")
        f.write(f"  Average Area [norm²]:          {np.mean([m['area'] for m in pelog]):>8.4f}\n")
        f.write(f"  Average Path Length [norm]:    {np.mean([m['path_length'] for m in pelog]):>8.3f}\n")
        f.write(f"  Average Revisit Rate [-]:      {np.mean([m['revisit_rate'] for m in pelog]):>8.3f}\n")
        f.write(f"  Average Correlation [-]:       {np.mean([m['correlation'] for m in pelog]):>8.3f}\n\n")
        
        f.write(f"Slendro (n={len(slendro)}):\n")
        f.write(f"  Average Area [norm²]:          {np.mean([m['area'] for m in slendro]):>8.4f}\n")
        f.write(f"  Average Path Length [norm]:    {np.mean([m['path_length'] for m in slendro]):>8.3f}\n")
        f.write(f"  Average Revisit Rate [-]:      {np.mean([m['revisit_rate'] for m in slendro]):>8.3f}\n")
        f.write(f"  Average Correlation [-]:       {np.mean([m['correlation'] for m in slendro]):>8.3f}\n\n")
        
        # Comparison by composition type
        f.write("COMPARISON BY COMPOSITION TYPE\n")
        f.write("-"*100 + "\n\n")
        
        for comp_type in ['layered', 'alternating', 'melodic', 'spectral']:
            subset = [m for m in all_metrics if comp_type in m['mode']]
            if len(subset) > 0:
                f.write(f"{comp_type.capitalize()} (n={len(subset)}):\n")
                f.write(f"  Average Area [norm²]:             {np.mean([m['area'] for m in subset]):>8.4f}\n")
                f.write(f"  Average Exploration Index [norm]: {np.mean([m['exploration_index'] for m in subset]):>8.4f}\n")
                f.write(f"  Average Revisit Rate [-]:         {np.mean([m['revisit_rate'] for m in subset]):>8.3f}\n\n")
        
        # Detailed mode profiles
        f.write("\n" + "="*100 + "\n")
        f.write("DETAILED MODE PROFILES\n")
        f.write("="*100 + "\n\n")
        
        for m in all_metrics:
            f.write(f"MODE: {m['mode'].upper()}\n")
            f.write("-"*100 + "\n")
            f.write(f"Phase Space Coverage:\n")
            f.write(f"  Area (convex hull) [norm²]:   {m['area']:>10.6f}\n")
            f.write(f"  Path Length [norm]:           {m['path_length']:>10.3f}\n")
            f.write(f"  Exploration Index [norm]:     {m['exploration_index']:>10.6f}\n\n")
            
            f.write(f"Trajectory Centroid [normalized]:\n")
            f.write(f"  Brightness:                   {m['centroid_brightness']:>10.6f}\n")
            f.write(f"  Energy:                       {m['centroid_energy']:>10.6f}\n\n")
            
            f.write(f"Trajectory Spread [normalized]:\n")
            f.write(f"  Brightness (std):             {m['spread_brightness']:>10.6f}\n")
            f.write(f"  Energy (std):                 {m['spread_energy']:>10.6f}\n\n")
            
            f.write(f"Dynamical Properties:\n")
            f.write(f"  Brightness-Energy Corr [-]:   {m['correlation']:>10.6f} (p={m['correlation_pvalue']:.6f})\n")
            f.write(f"  Revisit Rate [-]:             {m['revisit_rate']:>10.6f}\n\n")
            
            # Interpretation
            f.write(f"Interpretation:\n")
            
            if m['area'] > 0.1:
                f.write(f"  Coverage:    Large - Explores wide range of musical states\n")
            elif m['area'] > 0.05:
                f.write(f"  Coverage:    Moderate - Balanced exploration\n")
            else:
                f.write(f"  Coverage:    Small - Concentrated in specific region\n")
            
            if m['revisit_rate'] > 0.3:
                f.write(f"  Pattern:     High revisit - Cyclic, recurring structures\n")
            elif m['revisit_rate'] > 0.15:
                f.write(f"  Pattern:     Moderate revisit - Some recurring elements\n")
            else:
                f.write(f"  Pattern:     Low revisit - Progressive, non-repeating\n")
            
            if abs(m['correlation']) > 0.5:
                direction = "positive" if m['correlation'] > 0 else "negative"
                f.write(f"  Coupling:    Strong {direction} - Brightness and energy move together\n")
            elif abs(m['correlation']) > 0.3:
                f.write(f"  Coupling:    Moderate - Some coordination between dimensions\n")
            else:
                f.write(f"  Coupling:    Weak - Independent variation in brightness and energy\n")
            
            f.write("\n")
        
        f.write("="*100 + "\n")
        f.write("UNITS SUMMARY\n")
        f.write("="*100 + "\n")
        f.write("norm     = normalized to 0-1 scale [dimensionless]\n")
        f.write("norm²    = square of normalized units [dimensionless]\n")
        f.write("[-]      = dimensionless ratio or correlation coefficient\n")
        f.write("="*100 + "\n")
        f.write("END OF PHASE SPACE ANALYSIS\n")
        f.write("="*100 + "\n")


def write_metrics_csv(all_metrics, output_file):
    """Write phase space metrics to CSV."""
    if len(all_metrics) == 0:
        return
    
    fieldnames = list(all_metrics[0].keys())
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("PHASE SPACE ANALYSIS")
    print("="*70 + "\n")
    
    FIGS_DIR.mkdir(exist_ok=True)
    STATS_DIR.mkdir(exist_ok=True)
    
    # Check if state variable files exist
    print("Checking for state variable files...")
    all_modes = PELOG_MODES + SLENDRO_MODES
    available_modes = [m for m in all_modes if (OUTPUT_DIR / f'state_variables_{m}.csv').exists()]
    
    if len(available_modes) == 0:
        print("\nERROR: No state variable files found!")
        print(f"Expected location: {OUTPUT_DIR}/")
        print("Please run music_characterization_revised.py first.")
        return
    
    print(f"  Found {len(available_modes)}/{len(all_modes)} mode files\n")
    
    # Calculate phase space metrics for all modes
    print("Calculating phase space metrics...")
    all_metrics = []
    
    for mode in available_modes:
        data_file = OUTPUT_DIR / f'state_variables_{mode}.csv'
        df = pd.read_csv(data_file)
        metrics = analyze_phase_space(mode, df)
        all_metrics.append(metrics)
        print(f"  ✓ {mode}")
    
    # Create plots
    print("\nCreating phase space plots...")
    
    # Pelog plot
    pelog_available = [m for m in PELOG_MODES if m in available_modes]
    if len(pelog_available) == 4:
        pelog_output = FIGS_DIR / 'fig_phase_space_pelog.png'
        create_phase_space_plot(pelog_available, 'Pelog', pelog_output)
    else:
        print(f"  ! Skipping Pelog plot (only {len(pelog_available)}/4 modes available)")
    
    # Slendro plot
    slendro_available = [m for m in SLENDRO_MODES if m in available_modes]
    if len(slendro_available) == 4:
        slendro_output = FIGS_DIR / 'fig_phase_space_slendro.png'
        create_phase_space_plot(slendro_available, 'Slendro', slendro_output)
    else:
        print(f"  ! Skipping Slendro plot (only {len(slendro_available)}/4 modes available)")
    
    # Write analysis report
    print("\nWriting analysis results...")
    
    report_file = STATS_DIR / 'phase_space_analysis.txt'
    write_phase_space_report(all_metrics, report_file)
    print(f"  ✓ {report_file.name}")
    
    csv_file = STATS_DIR / 'phase_space_metrics.csv'
    write_metrics_csv(all_metrics, csv_file)
    print(f"  ✓ {csv_file.name}")
    
    # Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    print(f"\nFigures saved to:  {FIGS_DIR}/")
    print(f"Analysis saved to: {STATS_DIR}/")
    
    print("\nPhase Space Insights:")
    
    # Find interesting modes
    max_area = max(all_metrics, key=lambda x: x['area'])
    max_revisit = max(all_metrics, key=lambda x: x['revisit_rate'])
    max_exploration = max(all_metrics, key=lambda x: x['exploration_index'])
    
    print(f"  Largest coverage:     {max_area['mode']:<25} (area = {max_area['area']:.4f} norm²)")
    print(f"  Most cyclic:          {max_revisit['mode']:<25} (revisit = {max_revisit['revisit_rate']:.3f})")
    print(f"  Most exploratory:     {max_exploration['mode']:<25} (index = {max_exploration['exploration_index']:.4f} norm)")
    
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
