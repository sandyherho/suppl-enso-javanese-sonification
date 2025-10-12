#!/usr/bin/env python
"""
music_analysis_complete.py

Complete Music Analysis Pipeline
=================================

Combines correlation analysis and musical feature extraction in one script.

Outputs:
1. Analysis results (stats/ directory):
   - music_analysis_results.pkl
   - music_analysis_results.csv
   - music_analysis_results.txt

2. Time series (output_data/ directory):
   - music_timeseries_pelog_layered.csv
   - music_timeseries_pelog_alternating.csv
   ... (one per mode)
   - README.txt

Authors: Sandy H. S. Herho, Rusmawan Suwarman, Edi Riawan, Nurjanna J. Trilaksono
Institution: Weather and Climate Prediction Laboratory (WCPL) ITB
Date: 10/12/2025
License: WTFPL
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from mido import MidiFile
from multiprocessing import Pool, cpu_count
import pickle
import csv
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')

# Configuration
INPUT_DIR = Path('../input_data')
MUSIC_DIR = Path('../music_outputs')
STATS_DIR = Path('../stats')
OUTPUT_DIR = Path('../output_data')

INPUT_FILE = INPUT_DIR / 'nino34_hadisst_mon_1870_2024.csv'

MODES = ['pelog_layered', 'pelog_alternating', 'pelog_melodic', 'pelog_spectral',
         'slendro_layered', 'slendro_alternating', 'slendro_melodic', 'slendro_spectral']


# ============================================================================
# PART 1: CORRELATION ANALYSIS
# ============================================================================

def load_enso_data(filepath):
    """Load ENSO Nino 3.4 data."""
    df = pd.read_csv(filepath, skipinitialspace=True)
    df.columns = df.columns.str.strip().str.lower()
    df['date'] = pd.to_datetime(df['date'])
    df['nino34'] = pd.to_numeric(df['nino34'], errors='coerce')
    df.loc[df['nino34'] < -900, 'nino34'] = np.nan
    df = df.dropna(subset=['nino34'])
    return df['nino34'].values


def find_optimal_lag(x, y, max_lag=12):
    """Find optimal time lag between two signals."""
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]
    
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    
    if len(x) < max_lag + 10:
        return 0, 0.0
    
    correlations = []
    lags = range(-max_lag, max_lag + 1)
    
    for lag in lags:
        if lag < 0:
            corr = np.corrcoef(x[:lag], y[-lag:])[0, 1] if len(x[:lag]) > 1 else 0.0
        elif lag > 0:
            corr = np.corrcoef(x[lag:], y[:-lag])[0, 1] if len(x[lag:]) > 1 else 0.0
        else:
            corr = np.corrcoef(x, y)[0, 1]
        
        correlations.append(corr if np.isfinite(corr) else 0.0)
    
    correlations = np.array(correlations)
    optimal_idx = np.argmax(np.abs(correlations))
    optimal_lag = lags[optimal_idx]
    max_correlation = correlations[optimal_idx]
    
    return optimal_lag, max_correlation


def pearson_correlation(x, y):
    """Calculate Pearson correlation coefficient."""
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]
    
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    
    if len(x) < 2:
        return 0.0, 1.0
    
    try:
        corr, pval = stats.pearsonr(x, y)
        return corr, pval
    except:
        return 0.0, 1.0


def spearman_correlation(x, y):
    """Calculate Spearman rank correlation."""
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]
    
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    
    if len(x) < 2:
        return 0.0, 1.0
    
    try:
        corr, pval = stats.spearmanr(x, y)
        return corr, pval
    except:
        return 0.0, 1.0


def phase_separation(pitches, enso_data):
    """Calculate how well music separates El Niño from La Niña."""
    min_len = min(len(pitches), len(enso_data))
    pitches = pitches[:min_len]
    enso_data = enso_data[:min_len]
    
    el_nino = enso_data >= 0.5
    la_nina = enso_data <= -0.5
    
    if np.sum(el_nino) < 2 or np.sum(la_nina) < 2:
        return 0.0
    
    mean_el_nino = np.mean(pitches[el_nino])
    mean_la_nina = np.mean(pitches[la_nina])
    
    std_el_nino = np.std(pitches[el_nino])
    std_la_nina = np.std(pitches[la_nina])
    pooled_std = np.sqrt((std_el_nino**2 + std_la_nina**2) / 2)
    
    if pooled_std == 0:
        return 0.0
    
    cohens_d = abs(mean_el_nino - mean_la_nina) / pooled_std
    
    return cohens_d


def directional_agreement(x, y):
    """Calculate percentage of time series changes that agree in direction."""
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]
    
    if len(x) < 10:
        return 0.0
    
    x_changes = np.diff(x)
    y_changes = np.diff(y)
    
    agreements = np.sign(x_changes) == np.sign(y_changes)
    moving = (x_changes != 0) & (y_changes != 0)
    
    if np.sum(moving) == 0:
        return 0.0
    
    percentage = 100 * np.sum(agreements[moving]) / np.sum(moving)
    
    return percentage


def load_midi_features(midi_file):
    """Extract features from MIDI file."""
    try:
        mid = MidiFile(midi_file)
        
        pitches = []
        velocities = []
        times = []
        
        current_time = 0
        for track in mid.tracks:
            for msg in track:
                current_time += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    pitches.append(msg.note)
                    velocities.append(msg.velocity)
                    times.append(current_time)
        
        if len(pitches) == 0:
            return None
            
        return {
            'pitches': np.array(pitches),
            'velocities': np.array(velocities),
            'times': np.array(times)
        }
    except Exception as e:
        print(f"    Error loading {midi_file.name}: {e}")
        return None


def analyze_single_mode(args):
    """Analyze one sonification mode."""
    mode_name, enso_data = args
    
    results = {
        'mode': mode_name,
        'midi_exists': False,
        'error': None
    }
    
    try:
        midi_file = MUSIC_DIR / f'enso_javanese_{mode_name}.mid'
        if midi_file.exists():
            midi_data = load_midi_features(midi_file)
            if midi_data is not None:
                results['midi_exists'] = True
                pitches = midi_data['pitches']
                velocities = midi_data['velocities']
                
                # Resample ENSO to match pitch length
                enso_resampled = np.interp(
                    np.linspace(0, len(enso_data)-1, len(pitches)),
                    np.arange(len(enso_data)), 
                    enso_data
                )
                
                # Correlation metrics
                corr, corr_pval = pearson_correlation(enso_resampled, pitches)
                results['correlation'] = float(corr)
                results['correlation_pvalue'] = float(corr_pval)
                
                spear, spear_pval = spearman_correlation(enso_resampled, pitches)
                results['spearman_correlation'] = float(spear)
                results['spearman_pvalue'] = float(spear_pval)
                
                # Lag analysis
                optimal_lag, lag_corr = find_optimal_lag(enso_resampled, pitches, max_lag=12)
                results['optimal_lag'] = int(optimal_lag)
                results['lag_correlation'] = float(lag_corr)
                
                # Phase separation
                results['phase_separation'] = float(phase_separation(pitches, enso_resampled))
                
                # Directional agreement
                results['directional_agreement'] = float(directional_agreement(enso_resampled, pitches))
                
                # Musical characteristics
                results['pitch_mean'] = float(np.mean(pitches))
                results['pitch_std'] = float(np.std(pitches))
                results['pitch_min'] = int(np.min(pitches))
                results['pitch_max'] = int(np.max(pitches))
                results['pitch_range'] = int(np.max(pitches) - np.min(pitches))
                results['unique_pitches'] = int(len(np.unique(pitches)))
                
                results['velocity_mean'] = float(np.mean(velocities))
                results['velocity_std'] = float(np.std(velocities))
                results['velocity_min'] = int(np.min(velocities))
                results['velocity_max'] = int(np.max(velocities))
                
                # ENSO-pitch relationship in different phases
                el_nino = enso_resampled >= 0.5
                la_nina = enso_resampled <= -0.5
                neutral = (~el_nino) & (~la_nina)
                
                if np.sum(el_nino) > 0:
                    results['pitch_mean_el_nino'] = float(np.mean(pitches[el_nino]))
                else:
                    results['pitch_mean_el_nino'] = np.nan
                
                if np.sum(la_nina) > 0:
                    results['pitch_mean_la_nina'] = float(np.mean(pitches[la_nina]))
                else:
                    results['pitch_mean_la_nina'] = np.nan
                
                if np.sum(neutral) > 0:
                    results['pitch_mean_neutral'] = float(np.mean(pitches[neutral]))
                else:
                    results['pitch_mean_neutral'] = np.nan
    
    except Exception as e:
        results['error'] = str(e)
        print(f"    Error analyzing {mode_name}: {e}")
    
    return results


# ============================================================================
# PART 2: PITCH CENTER OF MASS EXTRACTION
# ============================================================================

def extract_pitch_center_of_mass(midi_file, time_resolution=0.5):
    """
    Extract Pitch Center of Mass (PCM) time series from MIDI.
    
    PCM(t) = Σ[pitch_i × velocity_i] / Σ[velocity_i]
    """
    try:
        mid = MidiFile(midi_file)
        
        # Get tempo
        tempo = 500000  # Default: 120 BPM
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                    break
        
        # Extract all note events
        note_events = []
        
        for track in mid.tracks:
            current_time = 0
            for msg in track:
                current_time += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    note_events.append({
                        'time': current_time,
                        'pitch': msg.note,
                        'velocity': msg.velocity,
                        'type': 'on'
                    })
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    note_events.append({
                        'time': current_time,
                        'pitch': msg.note,
                        'velocity': 0,
                        'type': 'off'
                    })
        
        if len(note_events) == 0:
            return None
        
        # Sort by time
        note_events.sort(key=lambda x: x['time'])
        
        # Get time range
        max_time = note_events[-1]['time']
        ticks_per_beat = mid.ticks_per_beat
        max_beats = max_time / ticks_per_beat
        
        # Create time grid
        num_steps = int(np.ceil(max_beats / time_resolution))
        time_grid = np.arange(0, num_steps) * time_resolution
        
        # Track active notes at each time point
        pcm_series = []
        velocity_series = []
        density_series = []
        variance_series = []
        
        active_notes = {}  # {pitch: velocity}
        event_idx = 0
        
        for t in time_grid:
            t_ticks = t * ticks_per_beat
            
            # Process all events up to current time
            while event_idx < len(note_events) and note_events[event_idx]['time'] <= t_ticks:
                event = note_events[event_idx]
                
                if event['type'] == 'on':
                    active_notes[event['pitch']] = event['velocity']
                else:
                    active_notes.pop(event['pitch'], None)
                
                event_idx += 1
            
            # Calculate Pitch Center of Mass
            if len(active_notes) > 0:
                pitches = np.array(list(active_notes.keys()))
                velocities = np.array(list(active_notes.values()))
                
                pcm = np.sum(pitches * velocities) / np.sum(velocities)
                vel_mean = np.mean(velocities)
                density = len(active_notes)
                variance = np.var(pitches)
                
            else:
                pcm = pcm_series[-1] if len(pcm_series) > 0 else 60.0
                vel_mean = velocity_series[-1] if len(velocity_series) > 0 else 0.0
                density = 0
                variance = 0.0
            
            pcm_series.append(pcm)
            velocity_series.append(vel_mean)
            density_series.append(density)
            variance_series.append(variance)
        
        # Create DataFrame
        df = pd.DataFrame({
            'time_step': np.arange(len(time_grid)),
            'time_beats': time_grid,
            'pitch_center_mass': pcm_series,
            'velocity_mean': velocity_series,
            'note_density': density_series,
            'pitch_variance': variance_series
        })
        
        return df
    
    except Exception as e:
        print(f"    Error processing {midi_file.name}: {e}")
        return None


def smooth_timeseries(data, window_size=5):
    """Apply light smoothing to remove high-frequency jitter."""
    if len(data) < window_size:
        return data
    
    smoothed = pd.Series(data).rolling(
        window=window_size, 
        center=True, 
        min_periods=1
    ).mean().values
    
    return smoothed


# ============================================================================
# OUTPUT WRITERS
# ============================================================================

def write_csv(all_results, output_file):
    """Write analysis results to CSV."""
    with open(output_file, 'w', newline='') as f:
        fieldnames = [
            'mode', 'midi_exists', 'error',
            'correlation', 'correlation_pvalue',
            'spearman_correlation', 'spearman_pvalue',
            'optimal_lag', 'lag_correlation',
            'phase_separation', 'directional_agreement',
            'pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max', 'pitch_range', 'unique_pitches',
            'velocity_mean', 'velocity_std', 'velocity_min', 'velocity_max',
            'pitch_mean_el_nino', 'pitch_mean_la_nina', 'pitch_mean_neutral'
        ]
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in all_results:
            row = {k: r.get(k, '') for k in fieldnames}
            writer.writerow(row)


def write_text_report(all_results, output_file):
    """Write descriptive text report."""
    with open(output_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("GAMELAN SONIFICATION ANALYSIS\n")
        f.write("="*100 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Authors: Sandy H. S. Herho, Rusmawan Suwarman, Edi Riawan, Nurjanna J. Trilaksono\n")
        f.write("Institution: Weather and Climate Prediction Laboratory (WCPL) ITB\n")
        f.write("="*100 + "\n\n")
        
        f.write("ANALYSIS APPROACH\n")
        f.write("-"*100 + "\n")
        f.write("This analysis uses straightforward statistical metrics to evaluate the relationship\n")
        f.write("between ENSO data and the gamelan music:\n\n")
        f.write("1. Correlation Analysis - Pearson and Spearman correlations\n")
        f.write("2. Lag Analysis - Optimal time shift between signals\n")
        f.write("3. Phase Separation - Cohen's d for El Niño vs La Niña distinction\n")
        f.write("4. Directional Agreement - Percentage of changes in same direction\n")
        f.write("5. Musical Characteristics - Pitch and velocity distributions\n")
        f.write("\n\n")
        
        # Summary table
        f.write("SUMMARY TABLE\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Mode':<25} {'Corr':<8} {'P-val':<10} {'Lag':<6} {'Phase':<8} {'Dir%':<8}\n")
        f.write("-"*100 + "\n")
        
        modes_with_data = [r for r in all_results if r['midi_exists']]
        
        for r in modes_with_data:
            corr = r.get('correlation', 0)
            pval = r.get('correlation_pvalue', 1)
            lag = r.get('optimal_lag', 0)
            phase = r.get('phase_separation', 0)
            dir_agree = r.get('directional_agreement', 0)
            
            f.write(f"{r['mode']:<25} {corr:>6.3f}  {pval:>8.5f}  {lag:>4d}    {phase:>6.3f}  {dir_agree:>6.1f}%\n")
        
        f.write("\n\n")
        
        # Detailed results per mode
        f.write("DETAILED RESULTS BY MODE\n")
        f.write("="*100 + "\n\n")
        
        for r in modes_with_data:
            f.write(f"MODE: {r['mode'].upper()}\n")
            f.write("-"*100 + "\n\n")
            
            f.write("Correlation Metrics:\n")
            corr = r.get('correlation', 0)
            pval = r.get('correlation_pvalue', 1)
            spear = r.get('spearman_correlation', 0)
            spear_pval = r.get('spearman_pvalue', 1)
            
            f.write(f"  Pearson correlation:         {corr:>7.4f}  (p = {pval:.6f})\n")
            
            if pval < 0.001:
                f.write(f"    Statistical significance: Very strong (p < 0.001)\n")
            elif pval < 0.01:
                f.write(f"    Statistical significance: Strong (p < 0.01)\n")
            elif pval < 0.05:
                f.write(f"    Statistical significance: Significant (p < 0.05)\n")
            elif pval < 0.1:
                f.write(f"    Statistical significance: Marginal (p < 0.1)\n")
            else:
                f.write(f"    Statistical significance: Not significant at α=0.1\n")
            
            f.write(f"  Spearman correlation:        {spear:>7.4f}  (p = {spear_pval:.6f})\n\n")
            
            f.write("Lag Analysis:\n")
            lag = r.get('optimal_lag', 0)
            lag_corr = r.get('lag_correlation', 0)
            f.write(f"  Optimal lag:                 {lag:>4d} time steps\n")
            f.write(f"  Correlation at optimal lag:  {lag_corr:>7.4f}\n\n")
            
            f.write("Phase Separation:\n")
            phase = r.get('phase_separation', 0)
            f.write(f"  Cohen's d:                   {phase:>7.4f}\n")
            
            if phase >= 0.8:
                f.write(f"    Interpretation: Large effect\n")
            elif phase >= 0.5:
                f.write(f"    Interpretation: Medium effect\n")
            elif phase >= 0.2:
                f.write(f"    Interpretation: Small effect\n")
            else:
                f.write(f"    Interpretation: Minimal effect\n")
            f.write("\n")
            
            f.write("Directional Agreement:\n")
            dir_agree = r.get('directional_agreement', 0)
            f.write(f"  Agreement percentage:        {dir_agree:>6.1f}%\n\n")
            
            f.write("Musical Characteristics:\n")
            f.write(f"  Pitch range:                 {r.get('pitch_range', 0):>4d} semitones\n")
            f.write(f"  Unique pitches:              {r.get('unique_pitches', 0):>4d}\n")
            f.write(f"  Pitch mean ± std:            {r.get('pitch_mean', 0):>6.2f} ± {r.get('pitch_std', 0):>5.2f}\n")
            f.write(f"  Velocity mean ± std:         {r.get('velocity_mean', 0):>6.2f} ± {r.get('velocity_std', 0):>5.2f}\n")
            f.write("\n")
            
            f.write("."*100 + "\n\n")
        
        f.write("="*100 + "\n")
        f.write("END OF ANALYSIS\n")
        f.write("="*100 + "\n")


def create_readme():
    """Create README explaining the output files."""
    readme_file = OUTPUT_DIR / 'README.txt'
    
    with open(readme_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MUSIC ANALYSIS OUTPUT DATA\n")
        f.write("="*80 + "\n\n")
        
        f.write("TIME SERIES FILES\n")
        f.write("-"*80 + "\n\n")
        
        f.write("Each music_timeseries_*.csv contains:\n\n")
        
        f.write("Columns:\n")
        f.write("  time_step                    : Sequential integer (0, 1, 2, ...)\n")
        f.write("  time_beats                   : Time in musical beats\n")
        f.write("  pitch_center_mass            : Weighted average pitch (raw)\n")
        f.write("  velocity_mean                : Average velocity of active notes\n")
        f.write("  note_density                 : Number of simultaneous notes\n")
        f.write("  pitch_variance               : Spread of pitches\n")
        f.write("  pitch_center_mass_smoothed   : Smoothed PCM (for plotting)\n")
        f.write("  pitch_center_mass_derivative : Rate of change of PCM\n\n")
        
        f.write("PRIMARY VARIABLE: pitch_center_mass_smoothed\n\n")
        
        f.write("MATHEMATICAL DEFINITION\n")
        f.write("-"*80 + "\n\n")
        
        f.write("Pitch Center of Mass (PCM):\n\n")
        f.write("  PCM(t) = Σ[pitch_i(t) × velocity_i(t)] / Σ[velocity_i(t)]\n\n")
        
        f.write("This represents the 'musical state' as a single continuous value,\n")
        f.write("analogous to center of mass in physics.\n\n")
        
        f.write("USAGE FOR PHASE SPACE ANALYSIS\n")
        f.write("-"*80 + "\n\n")
        
        f.write("Plot ENSO vs Musical State:\n")
        f.write("  plt.scatter(enso, music['pitch_center_mass_smoothed'])\n")
        f.write("  plt.xlabel('ENSO Index (°C)')\n")
        f.write("  plt.ylabel('Musical State (Pitch Center of Mass)')\n\n")
        
        f.write("="*80 + "\n")
        f.write("Generated: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
        f.write("="*80 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("COMPLETE MUSIC ANALYSIS PIPELINE")
    print("="*70 + "\n")
    
    # Create directories
    STATS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load ENSO data
    print("[1/3] Loading ENSO data...")
    enso_data = load_enso_data(INPUT_FILE)
    print(f"      Loaded {len(enso_data)} months of data\n")
    
    # Run correlation analysis
    print("[2/3] Analyzing correlations...")
    n_cores = min(cpu_count(), len(MODES))
    print(f"      Using {n_cores} CPU cores")
    
    args_list = [(mode, enso_data) for mode in MODES]
    
    with Pool(n_cores) as pool:
        all_results = pool.map(analyze_single_mode, args_list)
    
    modes_found = sum([1 for r in all_results if r['midi_exists']])
    print(f"      Successfully analyzed {modes_found} modes\n")
    
    # Save correlation results
    print("      Saving correlation results...")
    
    pkl_file = STATS_DIR / 'music_analysis_results.pkl'
    with open(pkl_file, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"        ✓ {pkl_file.name}")
    
    csv_file = STATS_DIR / 'music_analysis_results.csv'
    write_csv(all_results, csv_file)
    print(f"        ✓ {csv_file.name}")
    
    txt_file = STATS_DIR / 'music_analysis_results.txt'
    write_text_report(all_results, txt_file)
    print(f"        ✓ {txt_file.name}")
    
    # Extract time series
    print("\n[3/3] Extracting musical time series...")
    
    extracted_count = 0
    
    for mode in MODES:
        midi_file = MUSIC_DIR / f'enso_javanese_{mode}.mid'
        
        if not midi_file.exists():
            continue
        
        print(f"      Processing: {mode}")
        
        df = extract_pitch_center_of_mass(midi_file, time_resolution=0.5)
        
        if df is None:
            print(f"        ✗ Failed")
            continue
        
        # Add smoothed version and derivative
        df['pitch_center_mass_smoothed'] = smooth_timeseries(
            df['pitch_center_mass'].values, 
            window_size=5
        )
        
        df['pitch_center_mass_derivative'] = np.gradient(
            df['pitch_center_mass_smoothed'].values
        )
        
        # Save to CSV
        output_file = OUTPUT_DIR / f'music_timeseries_{mode}.csv'
        df.to_csv(output_file, index=False, float_format='%.6f')
        
        print(f"        ✓ {output_file.name} ({len(df)} time steps)")
        
        extracted_count += 1
    
    # Create README
    print("\n      Creating README...")
    create_readme()
    print(f"        ✓ README.txt")
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    
    print("\nCorrelation Analysis Results:")
    print(f"  Directory: {STATS_DIR.absolute()}")
    print(f"  Files: 3 (PKL, CSV, TXT)")
    
    print("\nTime Series Data:")
    print(f"  Directory: {OUTPUT_DIR.absolute()}")
    print(f"  Files: {extracted_count} time series + README")
    
    print("\nPrimary variable for phase space analysis:")
    print("  → pitch_center_mass_smoothed")
    
    # Quick summary
    if modes_found > 0:
        print("\n" + "="*70)
        print("QUICK SUMMARY")
        print("="*70 + "\n")
        
        modes_with_data = [r for r in all_results if r['midi_exists']]
        
        print(f"{'Mode':<30} {'Correlation':<12} {'P-value':<12}")
        print("-"*70)
        
        for r in modes_with_data:
            corr = r.get('correlation', 0)
            pval = r.get('correlation_pvalue', 1)
            print(f"{r['mode']:<30} {corr:>10.4f}  {pval:>10.6f}")
        
        all_corrs = [r['correlation'] for r in modes_with_data]
        print(f"\nAverage correlation: {np.mean(all_corrs):.4f}")
        
        sig_count = sum([1 for r in modes_with_data if r['correlation_pvalue'] < 0.05])
        print(f"Significant (p<0.05): {sig_count}/{len(modes_with_data)}")
    
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
