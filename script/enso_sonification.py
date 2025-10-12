#!/usr/bin/env python
"""
music_analysis_complete.py

Complete Music Analysis Pipeline with Proper Time Alignment
===========================================================

Uses novel pattern-based metrics instead of just correlation:
1. Dynamic Time Warping (DTW) distance
2. Event synchronization (El Niño/La Niña events vs musical peaks)
3. Phase coherence (frequency domain alignment)
4. Pattern recurrence analysis

Also generates aligned time series for direct phase space plotting.

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
from scipy import stats, signal
from scipy.spatial.distance import euclidean
from mido import MidiFile
from multiprocessing import Pool, cpu_count
import pickle
import csv
from datetime import datetime

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
# DATA LOADING
# ============================================================================

def load_enso_data(filepath):
    """Load ENSO Nino 3.4 data."""
    df = pd.read_csv(filepath, skipinitialspace=True)
    df.columns = df.columns.str.strip().str.lower()
    df['date'] = pd.to_datetime(df['date'])
    df['nino34'] = pd.to_numeric(df['nino34'], errors='coerce')
    df.loc[df['nino34'] < -900, 'nino34'] = np.nan
    df = df.dropna(subset=['nino34'])
    return df['date'].values, df['nino34'].values


# ============================================================================
# PITCH CENTER OF MASS EXTRACTION
# ============================================================================

def extract_pitch_center_of_mass(midi_file, time_resolution=0.5):
    """Extract Pitch Center of Mass (PCM) time series from MIDI."""
    try:
        mid = MidiFile(midi_file)
        
        tempo = 500000
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                    break
        
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
        
        note_events.sort(key=lambda x: x['time'])
        
        max_time = note_events[-1]['time']
        ticks_per_beat = mid.ticks_per_beat
        max_beats = max_time / ticks_per_beat
        
        num_steps = int(np.ceil(max_beats / time_resolution))
        time_grid = np.arange(0, num_steps) * time_resolution
        
        pcm_series = []
        velocity_series = []
        density_series = []
        
        active_notes = {}
        event_idx = 0
        
        for t in time_grid:
            t_ticks = t * ticks_per_beat
            
            while event_idx < len(note_events) and note_events[event_idx]['time'] <= t_ticks:
                event = note_events[event_idx]
                
                if event['type'] == 'on':
                    active_notes[event['pitch']] = event['velocity']
                else:
                    active_notes.pop(event['pitch'], None)
                
                event_idx += 1
            
            if len(active_notes) > 0:
                pitches = np.array(list(active_notes.keys()))
                velocities = np.array(list(active_notes.values()))
                
                pcm = np.sum(pitches * velocities) / np.sum(velocities)
                vel_mean = np.mean(velocities)
                density = len(active_notes)
            else:
                pcm = pcm_series[-1] if len(pcm_series) > 0 else 60.0
                vel_mean = velocity_series[-1] if len(velocity_series) > 0 else 0.0
                density = 0
            
            pcm_series.append(pcm)
            velocity_series.append(vel_mean)
            density_series.append(density)
        
        return np.array(pcm_series), np.array(velocity_series), np.array(density_series)
    
    except Exception as e:
        print(f"    Error processing {midi_file.name}: {e}")
        return None, None, None


# ============================================================================
# NOVEL PATTERN-BASED METRICS
# ============================================================================

def dtw_distance(x, y, window=50):
    """
    Dynamic Time Warping distance - measures pattern similarity
    allowing for temporal stretching/compression.
    
    Lower = more similar patterns
    """
    n, m = len(x), len(y)
    
    # Normalize
    x = (x - np.mean(x)) / (np.std(x) + 1e-10)
    y = (y - np.mean(y)) / (np.std(y) + 1e-10)
    
    # DTW with window constraint (Sakoe-Chiba band)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(max(1, i - window), min(m + 1, i + window)):
            cost = (x[i-1] - y[j-1])**2
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    
    return np.sqrt(dtw[n, m])


def event_synchronization(enso, music, threshold_enso=1.0, threshold_music=None):
    """
    Event synchronization - measures how well extreme ENSO events
    align with musical peaks/troughs.
    
    Returns proportion of ENSO events that have musical events nearby.
    Range: 0-1, higher = better synchronization
    """
    if threshold_music is None:
        threshold_music = np.std(music)
    
    # Find ENSO events (El Niño and La Niña)
    enso_events = np.where(np.abs(enso) >= threshold_enso)[0]
    
    # Find musical events (peaks and troughs)
    music_peaks, _ = signal.find_peaks(music, prominence=threshold_music)
    music_troughs, _ = signal.find_peaks(-music, prominence=threshold_music)
    music_events = np.sort(np.concatenate([music_peaks, music_troughs]))
    
    if len(enso_events) == 0 or len(music_events) == 0:
        return 0.0
    
    # For each ENSO event, check if there's a music event within window
    window = 3  # Allow ±3 time steps
    synchronized = 0
    
    for enso_t in enso_events:
        if np.any(np.abs(music_events - enso_t) <= window):
            synchronized += 1
    
    return synchronized / len(enso_events)


def phase_coherence(x, y):
    """
    Phase coherence - measures frequency domain alignment.
    
    Uses cross-spectral density to find coherence.
    Range: 0-1, higher = better coherence
    """
    if len(x) < 50 or len(y) < 50:
        return 0.0
    
    try:
        f, Cxy = signal.coherence(x, y, nperseg=min(256, len(x)//4))
        
        # Average coherence across frequencies
        mean_coherence = np.mean(Cxy)
        
        return mean_coherence
    except:
        return 0.0


def pattern_recurrence(x, y, embedding_dim=3, threshold=0.5):
    """
    Recurrence-based similarity - measures if similar patterns
    recur in both signals.
    
    Uses time-delay embedding and recurrence analysis.
    Range: 0-1, higher = more similar recurrence patterns
    """
    n = min(len(x), len(y))
    if n < embedding_dim * 10:
        return 0.0
    
    x = x[:n]
    y = y[:n]
    
    # Normalize
    x = (x - np.mean(x)) / (np.std(x) + 1e-10)
    y = (y - np.mean(y)) / (np.std(y) + 1e-10)
    
    # Time-delay embedding (create phase space vectors)
    def embed(signal, dim):
        n = len(signal) - dim + 1
        embedded = np.zeros((n, dim))
        for i in range(n):
            embedded[i] = signal[i:i+dim]
        return embedded
    
    x_embed = embed(x, embedding_dim)
    y_embed = embed(y, embedding_dim)
    
    n_embed = min(len(x_embed), len(y_embed))
    
    # Sample to avoid O(n²) computation
    sample_size = min(500, n_embed)
    indices = np.random.choice(n_embed, sample_size, replace=False)
    
    x_sample = x_embed[indices]
    y_sample = y_embed[indices]
    
    # Calculate recurrence similarity
    similarity_count = 0
    
    for i in range(len(x_sample)):
        # Find neighbors in x
        x_dist = np.linalg.norm(x_sample - x_sample[i], axis=1)
        x_neighbors = x_dist < threshold
        
        # Find neighbors in y
        y_dist = np.linalg.norm(y_sample - y_sample[i], axis=1)
        y_neighbors = y_dist < threshold
        
        # Check overlap
        overlap = np.sum(x_neighbors & y_neighbors)
        similarity_count += overlap
    
    max_possible = len(x_sample) * len(x_sample)
    
    return similarity_count / max_possible if max_possible > 0 else 0.0


# ============================================================================
# ANALYSIS FUNCTION
# ============================================================================

def analyze_single_mode(args):
    """Analyze one sonification mode with novel metrics."""
    mode_name, enso_dates, enso_data = args
    
    results = {
        'mode': mode_name,
        'success': False,
        'error': None
    }
    
    try:
        midi_file = MUSIC_DIR / f'enso_javanese_{mode_name}.mid'
        if not midi_file.exists():
            results['error'] = 'MIDI file not found'
            return results
        
        # Extract PCM
        pcm, velocity, density = extract_pitch_center_of_mass(midi_file, time_resolution=0.5)
        
        if pcm is None:
            results['error'] = 'Failed to extract PCM'
            return results
        
        # Smooth PCM
        pcm_smooth = pd.Series(pcm).rolling(window=5, center=True, min_periods=1).mean().values
        
        # CRITICAL: Resample music to exactly match ENSO length
        enso_len = len(enso_data)
        music_len = len(pcm_smooth)
        
        # Interpolate music to ENSO length
        pcm_aligned = np.interp(
            np.linspace(0, music_len - 1, enso_len),
            np.arange(music_len),
            pcm_smooth
        )
        
        velocity_aligned = np.interp(
            np.linspace(0, music_len - 1, enso_len),
            np.arange(music_len),
            velocity
        )
        
        # Now both have exact same length
        assert len(pcm_aligned) == len(enso_data), "Length mismatch!"
        
        results['success'] = True
        results['n_timesteps'] = enso_len
        
        # Save aligned time series for phase space plotting
        aligned_df = pd.DataFrame({
            'time_step': np.arange(enso_len),
            'date': enso_dates,
            'enso_nino34': enso_data,
            'pitch_center_mass': pcm_aligned,
            'velocity_mean': velocity_aligned
        })
        
        output_file = OUTPUT_DIR / f'aligned_timeseries_{mode_name}.csv'
        aligned_df.to_csv(output_file, index=False, float_format='%.6f')
        results['output_file'] = str(output_file.name)
        
        # Calculate novel metrics
        print(f"      Calculating pattern-based metrics...")
        
        # 1. DTW distance (lower = better)
        results['dtw_distance'] = float(dtw_distance(enso_data, pcm_aligned, window=50))
        
        # 2. Event synchronization (0-1, higher = better)
        results['event_sync'] = float(event_synchronization(enso_data, pcm_aligned))
        
        # 3. Phase coherence (0-1, higher = better)
        results['phase_coherence'] = float(phase_coherence(enso_data, pcm_aligned))
        
        # 4. Pattern recurrence (0-1, higher = better)
        results['pattern_recurrence'] = float(pattern_recurrence(enso_data, pcm_aligned))
        
        # Also keep traditional metrics for reference
        corr, pval = stats.pearsonr(enso_data, pcm_aligned)
        results['correlation'] = float(corr)
        results['correlation_pvalue'] = float(pval)
        
        # Musical characteristics
        results['pitch_mean'] = float(np.mean(pcm_aligned))
        results['pitch_std'] = float(np.std(pcm_aligned))
        results['pitch_range'] = float(np.max(pcm_aligned) - np.min(pcm_aligned))
        
        # Phase-specific analysis
        el_nino = enso_data >= 0.5
        la_nina = enso_data <= -0.5
        
        if np.sum(el_nino) > 0:
            results['pitch_mean_el_nino'] = float(np.mean(pcm_aligned[el_nino]))
        else:
            results['pitch_mean_el_nino'] = np.nan
        
        if np.sum(la_nina) > 0:
            results['pitch_mean_la_nina'] = float(np.mean(pcm_aligned[la_nina]))
        else:
            results['pitch_mean_la_nina'] = np.nan
        
        if not np.isnan(results['pitch_mean_el_nino']) and not np.isnan(results['pitch_mean_la_nina']):
            # Cohen's d
            pooled_std = np.sqrt((np.var(pcm_aligned[el_nino]) + np.var(pcm_aligned[la_nina])) / 2)
            if pooled_std > 0:
                results['cohens_d'] = float(abs(results['pitch_mean_el_nino'] - results['pitch_mean_la_nina']) / pooled_std)
            else:
                results['cohens_d'] = 0.0
        else:
            results['cohens_d'] = 0.0
        
    except Exception as e:
        results['error'] = str(e)
        print(f"    Error: {e}")
    
    return results


# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

def write_csv(all_results, output_file):
    """Write results to CSV."""
    with open(output_file, 'w', newline='') as f:
        fieldnames = [
            'mode', 'success', 'error', 'n_timesteps', 'output_file',
            'dtw_distance', 'event_sync', 'phase_coherence', 'pattern_recurrence',
            'correlation', 'correlation_pvalue', 'cohens_d',
            'pitch_mean', 'pitch_std', 'pitch_range',
            'pitch_mean_el_nino', 'pitch_mean_la_nina'
        ]
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in all_results:
            row = {k: r.get(k, '') for k in fieldnames}
            writer.writerow(row)


def write_text_report(all_results, output_file):
    """Write comprehensive text report."""
    with open(output_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("GAMELAN SONIFICATION ANALYSIS - PATTERN-BASED METRICS\n")
        f.write("="*100 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Authors: Sandy H. S. Herho, Rusmawan Suwarman, Edi Riawan, Nurjanna J. Trilaksono\n")
        f.write("Institution: Weather and Climate Prediction Laboratory (WCPL) ITB\n")
        f.write("="*100 + "\n\n")
        
        f.write("METHODOLOGY\n")
        f.write("-"*100 + "\n")
        f.write("This analysis uses novel pattern-based metrics instead of simple correlation:\n\n")
        
        f.write("1. Dynamic Time Warping (DTW) Distance\n")
        f.write("   - Measures pattern similarity allowing temporal warping\n")
        f.write("   - Lower values = more similar patterns\n")
        f.write("   - Scale: 0 (identical) to ~100+ (very different)\n\n")
        
        f.write("2. Event Synchronization\n")
        f.write("   - Measures alignment of ENSO extremes with musical peaks\n")
        f.write("   - Range: 0 (no sync) to 1 (perfect sync)\n\n")
        
        f.write("3. Phase Coherence\n")
        f.write("   - Frequency domain alignment of signals\n")
        f.write("   - Range: 0 (incoherent) to 1 (perfectly coherent)\n\n")
        
        f.write("4. Pattern Recurrence\n")
        f.write("   - Similar patterns recurring in both signals\n")
        f.write("   - Range: 0 (different patterns) to 1 (same patterns)\n\n")
        
        f.write("TIME ALIGNMENT: All music time series resampled to exactly match ENSO (1860 months)\n")
        f.write("\n\n")
        
        # Summary table
        f.write("SUMMARY TABLE\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Mode':<25} {'DTW':<10} {'Event':<10} {'Phase':<10} {'Recur':<10} {'Corr':<10}\n")
        f.write(f"{'':25} {'Dist':<10} {'Sync':<10} {'Coher':<10} {'':10} {'':10}\n")
        f.write("-"*100 + "\n")
        
        successful = [r for r in all_results if r['success']]
        
        for r in successful:
            dtw = r.get('dtw_distance', 0)
            evt = r.get('event_sync', 0)
            phs = r.get('phase_coherence', 0)
            rec = r.get('pattern_recurrence', 0)
            cor = r.get('correlation', 0)
            
            f.write(f"{r['mode']:<25} {dtw:>8.2f}  {evt:>8.4f}  {phs:>8.4f}  {rec:>8.4f}  {cor:>8.4f}\n")
        
        f.write("\n\n")
        
        # Detailed results
        f.write("DETAILED RESULTS BY MODE\n")
        f.write("="*100 + "\n\n")
        
        for r in successful:
            f.write(f"MODE: {r['mode'].upper()}\n")
            f.write("-"*100 + "\n\n")
            
            f.write("Pattern-Based Metrics:\n")
            f.write(f"  DTW Distance:                {r.get('dtw_distance', 0):>10.2f}\n")
            
            dtw = r.get('dtw_distance', 0)
            if dtw < 50:
                f.write(f"    Interpretation: Strong pattern similarity\n")
            elif dtw < 100:
                f.write(f"    Interpretation: Moderate pattern similarity\n")
            else:
                f.write(f"    Interpretation: Weak pattern similarity\n")
            
            f.write(f"\n  Event Synchronization:       {r.get('event_sync', 0):>10.4f}\n")
            evt = r.get('event_sync', 0)
            if evt >= 0.5:
                f.write(f"    Interpretation: Good alignment of extreme events\n")
            elif evt >= 0.3:
                f.write(f"    Interpretation: Moderate event alignment\n")
            else:
                f.write(f"    Interpretation: Weak event alignment\n")
            
            f.write(f"\n  Phase Coherence:             {r.get('phase_coherence', 0):>10.4f}\n")
            phs = r.get('phase_coherence', 0)
            if phs >= 0.5:
                f.write(f"    Interpretation: Strong frequency domain coupling\n")
            elif phs >= 0.3:
                f.write(f"    Interpretation: Moderate frequency coupling\n")
            else:
                f.write(f"    Interpretation: Weak frequency coupling\n")
            
            f.write(f"\n  Pattern Recurrence:          {r.get('pattern_recurrence', 0):>10.4f}\n")
            rec = r.get('pattern_recurrence', 0)
            if rec >= 0.3:
                f.write(f"    Interpretation: Similar recurrence patterns\n")
            elif rec >= 0.15:
                f.write(f"    Interpretation: Some pattern similarity\n")
            else:
                f.write(f"    Interpretation: Different recurrence patterns\n")
            
            f.write("\nTraditional Metrics (for reference):\n")
            f.write(f"  Pearson Correlation:         {r.get('correlation', 0):>10.4f}\n")
            f.write(f"  P-value:                     {r.get('correlation_pvalue', 1):>10.6f}\n")
            f.write(f"  Cohen's d:                   {r.get('cohens_d', 0):>10.4f}\n")
            
            f.write("\nMusical Characteristics:\n")
            f.write(f"  Pitch Mean:                  {r.get('pitch_mean', 0):>10.2f}\n")
            f.write(f"  Pitch Std:                   {r.get('pitch_std', 0):>10.2f}\n")
            f.write(f"  Pitch Range:                 {r.get('pitch_range', 0):>10.2f} semitones\n")
            
            if not np.isnan(r.get('pitch_mean_el_nino', np.nan)):
                f.write(f"\n  Mean Pitch (El Niño):        {r.get('pitch_mean_el_nino', 0):>10.2f}\n")
                f.write(f"  Mean Pitch (La Niña):        {r.get('pitch_mean_la_nina', 0):>10.2f}\n")
            
            f.write(f"\nAligned Time Series: {r.get('output_file', 'N/A')}\n")
            f.write(f"  Length: {r.get('n_timesteps', 0)} time steps (matched to ENSO)\n")
            
            f.write("\n" + "."*100 + "\n\n")
        
        f.write("="*100 + "\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("="*100 + "\n\n")
        
        f.write("Unlike simple correlation, these metrics capture different aspects of\n")
        f.write("relationship between ENSO and music:\n\n")
        
        f.write("- DTW: Captures pattern shape similarity even if temporally shifted\n")
        f.write("- Event Sync: Tests if climate extremes trigger musical extremes\n")
        f.write("- Phase Coherence: Measures cyclic/oscillatory alignment\n")
        f.write("- Pattern Recurrence: Checks if similar patterns repeat in both\n\n")
        
        f.write("For artistic sonification, these metrics are more meaningful than\n")
        f.write("simple linear correlation.\n\n")
        
        f.write("="*100 + "\n")
        f.write("END OF ANALYSIS\n")
        f.write("="*100 + "\n")


def create_readme():
    """Create README."""
    readme_file = OUTPUT_DIR / 'README.txt'
    
    with open(readme_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ALIGNED TIME SERIES FOR PHASE SPACE ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        f.write("CONTENTS:\n")
        f.write("  aligned_timeseries_*.csv - One file per sonification mode\n\n")
        
        f.write("FILE FORMAT:\n")
        f.write("  Each CSV contains exactly 1860 rows (matching ENSO data):\n\n")
        
        f.write("Columns:\n")
        f.write("  time_step         : Sequential index (0 to 1859)\n")
        f.write("  date              : Date (1870-01 to 2024-12)\n")
        f.write("  enso_nino34       : ENSO Nino 3.4 index (°C)\n")
        f.write("  pitch_center_mass : Musical state (aligned)\n")
        f.write("  velocity_mean     : Velocity (aligned)\n\n")
        
        f.write("USAGE FOR PHASE SPACE PLOTS:\n")
        f.write("  df = pd.read_csv('aligned_timeseries_pelog_layered.csv')\n")
        f.write("  plt.scatter(df['enso_nino34'], df['pitch_center_mass'])\n\n")
        
        f.write("All time series are perfectly aligned - no resampling needed!\n\n")
        
        f.write("="*80 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("COMPLETE MUSIC ANALYSIS - PATTERN-BASED METRICS")
    print("="*70 + "\n")
    
    STATS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("[1/2] Loading ENSO data...")
    enso_dates, enso_data = load_enso_data(INPUT_FILE)
    print(f"      ENSO length: {len(enso_data)} months\n")
    
    print("[2/2] Analyzing modes with pattern-based metrics...")
    print("      (This uses DTW, event sync, phase coherence, pattern recurrence)")
    
    results_list = []
    
    for mode in MODES:
        print(f"\n      Processing: {mode}")
        result = analyze_single_mode((mode, enso_dates, enso_data))
        results_list.append(result)
        
        if result['success']:
            print(f"        ✓ Success")
            print(f"          DTW distance: {result.get('dtw_distance', 0):.2f}")
            print(f"          Event sync: {result.get('event_sync', 0):.4f}")
            print(f"          Phase coherence: {result.get('phase_coherence', 0):.4f}")
            print(f"          Saved: {result.get('output_file', 'N/A')}")
        else:
            print(f"        ✗ Failed: {result.get('error', 'Unknown')}")
    
    # Save results
    print("\n      Saving analysis results...")
    
    pkl_file = STATS_DIR / 'music_analysis_results.pkl'
    with open(pkl_file, 'wb') as f:
        pickle.dump(results_list, f)
    print(f"        ✓ {pkl_file.name}")
    
    csv_file = STATS_DIR / 'music_analysis_results.csv'
    write_csv(results_list, csv_file)
    print(f"        ✓ {csv_file.name}")
    
    txt_file = STATS_DIR / 'music_analysis_results.txt'
    write_text_report(results_list, txt_file)
    print(f"        ✓ {txt_file.name}")
    
    # Create README
    print("\n      Creating README...")
    create_readme()
    print(f"        ✓ README.txt")
    
    # Summary
    successful = [r for r in results_list if r['success']]
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nSuccessfully analyzed: {len(successful)}/{len(MODES)} modes")
    print(f"\nOutput locations:")
    print(f"  Analysis results: {STATS_DIR.absolute()}")
    print(f"  Aligned time series: {OUTPUT_DIR.absolute()}")
    
    if len(successful) > 0:
        print("\n" + "="*70)
        print("QUICK SUMMARY (Pattern-Based Metrics)")
        print("="*70 + "\n")
        
        print(f"{'Mode':<30} {'DTW':<10} {'EventSync':<12} {'PhaseCoher':<12}")
        print("-"*70)
        
        for r in successful:
            dtw = r.get('dtw_distance', 0)
            evt = r.get('event_sync', 0)
            phs = r.get('phase_coherence', 0)
            print(f"{r['mode']:<30} {dtw:>8.2f}  {evt:>10.4f}  {phs:>10.4f}")
        
        avg_dtw = np.mean([r['dtw_distance'] for r in successful])
        avg_evt = np.mean([r['event_sync'] for r in successful])
        avg_phs = np.mean([r['phase_coherence'] for r in successful])
        
        print("-"*70)
        print(f"{'AVERAGE':<30} {avg_dtw:>8.2f}  {avg_evt:>10.4f}  {avg_phs:>10.4f}")
        
        print("\nInterpretation:")
        print(f"  Lower DTW = better pattern similarity")
        print(f"  Higher event sync = better extreme event alignment")
        print(f"  Higher phase coherence = better frequency coupling")
        
        print("\nAll time series now have EXACTLY {len(enso_data)} time steps!")
        print("Ready for direct phase space plotting with no resampling needed.")
    
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
