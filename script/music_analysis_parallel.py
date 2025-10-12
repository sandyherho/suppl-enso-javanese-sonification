#!/usr/bin/env python
"""
music_analysis_parallel.py

Fast Parallel Music Analysis with Fixed Binning
================================================

Authors: Sandy H. S. Herho, Rusmawan Suwarman, Edi Riawan, Nurjanna J. Trilaksono
Institution: Weather and Climate Prediction Laboratory (WCPL) ITB
Date: 10/11/2025
License: WTFPL
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import librosa
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

INPUT_FILE = INPUT_DIR / 'nino34_hadisst_mon_1870_2024.csv'
OUTPUT_PKL = STATS_DIR / 'music_analysis_results.pkl'
OUTPUT_TXT = STATS_DIR / 'music_analysis_results.txt'
OUTPUT_CSV = STATS_DIR / 'music_analysis_results.csv'

MODES = ['pelog_layered', 'pelog_alternating', 'pelog_melodic', 'pelog_spectral',
         'slendro_layered', 'slendro_alternating', 'slendro_melodic', 'slendro_spectral']


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
    """Calculate safe number of bins, handling edge cases."""
    if len(data) < 2:
        return min_bins
    
    # Remove any infinite or NaN values
    data_clean = data[np.isfinite(data)]
    if len(data_clean) < 2:
        return min_bins
    
    # Check if all values are the same
    if np.std(data_clean) == 0:
        return min_bins
    
    # Freedman-Diaconis rule
    try:
        iqr = np.percentile(data_clean, 75) - np.percentile(data_clean, 25)
        if iqr == 0:
            # Fall back to Sturges
            bins = int(np.ceil(np.log2(len(data_clean)) + 1))
        else:
            h = 2 * iqr / (len(data_clean)**(1/3))
            if h == 0:
                bins = default
            else:
                bins = int(np.ceil((np.max(data_clean) - np.min(data_clean)) / h))
    except:
        bins = default
    
    # Clamp to reasonable range
    bins = max(min_bins, min(bins, max_bins))
    
    return bins


def shannon_entropy(data):
    """Calculate Shannon entropy with safe binning."""
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


def permutation_entropy(data, order=3, delay=1):
    """Calculate Permutation Entropy."""
    n = len(data)
    if n < order * delay + 1:
        return 0.0
    
    permutations = {}
    
    for i in range(n - delay * (order - 1)):
        sorted_index_tuple = tuple(np.argsort([data[i + j * delay] for j in range(order)]))
        if sorted_index_tuple in permutations:
            permutations[sorted_index_tuple] += 1
        else:
            permutations[sorted_index_tuple] = 1
    
    total = sum(permutations.values())
    probabilities = [count / total for count in permutations.values()]
    
    pe = -sum([p * np.log2(p + 1e-10) for p in probabilities if p > 0])
    max_pe = np.log2(np.math.factorial(order))
    
    return pe / max_pe if max_pe > 0 else 0.0


def spectral_entropy(data):
    """Calculate Spectral Entropy."""
    from scipy.fft import fft
    
    if len(data) < 2:
        return 0.0
    
    ps = np.abs(fft(data))**2
    ps_norm = ps / (np.sum(ps) + 1e-10)
    ps_norm = ps_norm[ps_norm > 0]
    
    se = -np.sum(ps_norm * np.log2(ps_norm + 1e-10))
    max_se = np.log2(len(ps_norm))
    
    return se / max_se if max_se > 0 else 0.0


def mutual_information(x, y):
    """Calculate Mutual Information with safe binning."""
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]
    
    # Remove any infinite values
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    
    if len(x) < 2:
        return 0.0
    
    bins = min(safe_bins(np.concatenate([x, y])), 50)
    
    try:
        hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
        pxy = hist_2d / np.sum(hist_2d)
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        px_py = px[:, None] * py[None, :]
        
        nonzero = pxy > 0
        mi = np.sum(pxy[nonzero] * np.log2(pxy[nonzero] / (px_py[nonzero] + 1e-10) + 1e-10))
        
        return max(mi, 0.0)
    except:
        return 0.0


def normalized_mutual_information(x, y):
    """Calculate Normalized Mutual Information."""
    mi = mutual_information(x, y)
    hx, _ = shannon_entropy(x)
    hy, _ = shannon_entropy(y)
    
    if hx + hy == 0:
        return 0.0
    
    return 2 * mi / (hx + hy)


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


def load_wav_features(wav_file):
    """Extract audio features from WAV file."""
    try:
        y, sr = librosa.load(str(wav_file), sr=None, duration=60)  # Load only first 60s for speed
        
        features = {
            'spectral_centroid': librosa.feature.spectral_centroid(y=y, sr=sr)[0],
            'spectral_bandwidth': librosa.feature.spectral_bandwidth(y=y, sr=sr)[0],
            'zero_crossing_rate': librosa.feature.zero_crossing_rate(y)[0],
            'rms_energy': librosa.feature.rms(y=y)[0],
        }
        
        return features
    except Exception as e:
        print(f"    Error loading {wav_file.name}: {e}")
        return None


def analyze_single_mode(args):
    """Analyze one sonification mode (for parallel processing)."""
    mode_name, enso_data = args
    
    results = {
        'mode': mode_name,
        'midi_exists': False,
        'wav_exists': False,
        'error': None
    }
    
    try:
        # MIDI analysis
        midi_file = MUSIC_DIR / f'enso_javanese_{mode_name}.mid'
        if midi_file.exists():
            midi_data = load_midi_features(midi_file)
            if midi_data is not None:
                results['midi_exists'] = True
                pitches = midi_data['pitches']
                velocities = midi_data['velocities']
                
                results['pitch_entropy'], results['pitch_bins'] = shannon_entropy(pitches)
                results['velocity_entropy'], results['velocity_bins'] = shannon_entropy(velocities)
                results['pitch_sample_entropy'] = sample_entropy(pitches)
                results['pitch_approx_entropy'] = approximate_entropy(pitches)
                results['pitch_perm_entropy'] = permutation_entropy(pitches)
                results['pitch_spectral_entropy'] = spectral_entropy(pitches)
                
                # Resample ENSO to match pitch length
                enso_resampled = np.interp(
                    np.linspace(0, len(enso_data)-1, len(pitches)),
                    np.arange(len(enso_data)), 
                    enso_data
                )
                
                results['mi_enso_pitch'] = mutual_information(enso_resampled, pitches)
                results['nmi_enso_pitch'] = normalized_mutual_information(enso_resampled, pitches)
                results['mi_enso_velocity'] = mutual_information(enso_resampled, velocities)
        
        # WAV analysis
        wav_file = MUSIC_DIR / f'enso_javanese_{mode_name}.wav'
        if wav_file.exists():
            wav_features = load_wav_features(wav_file)
            if wav_features is not None:
                results['wav_exists'] = True
                results['spectral_centroid_entropy'], _ = shannon_entropy(wav_features['spectral_centroid'])
                results['zcr_entropy'], _ = shannon_entropy(wav_features['zero_crossing_rate'])
                
                enso_resampled_wav = np.interp(
                    np.linspace(0, len(enso_data)-1, len(wav_features['spectral_centroid'])),
                    np.arange(len(enso_data)), 
                    enso_data
                )
                results['mi_enso_spectral_centroid'] = mutual_information(
                    enso_resampled_wav, 
                    wav_features['spectral_centroid']
                )
    
    except Exception as e:
        results['error'] = str(e)
        print(f"    ✗ Error analyzing {mode_name}: {e}")
    
    return results


def write_results_txt(all_results, enso_data, output_file):
    """Write comprehensive results to text file."""
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MUSIC ANALYSIS RESULTS - INFORMATION-THEORETIC METRICS\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Authors: Sandy H. S. Herho, Rusmawan Suwarman, Edi Riawan, Nurjanna J. Trilaksono\n")
        f.write("Institution: Weather and Climate Prediction Laboratory (WCPL) ITB\n")
        f.write("="*80 + "\n\n")
        
        # ENSO baseline
        enso_se = sample_entropy(enso_data)
        enso_ae = approximate_entropy(enso_data)
        
        f.write("BASELINE ENSO SIGNAL CHARACTERISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Sample Entropy [dimensionless]:      {enso_se:.4f}\n")
        f.write(f"Approximate Entropy [dimensionless]:  {enso_ae:.4f}\n")
        f.write("\n")
        
        # Summary table
        f.write("SUMMARY - INFORMATION PRESERVATION BY MODE\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Mode':<25} {'Status':<10} {'MI [bits]':<12} {'NMI':<10} {'Preserve [%]':<12}\n")
        f.write("-"*80 + "\n")
        
        enso_entropy, _ = shannon_entropy(enso_data)
        
        for r in all_results:
            status = "✓ SUCCESS" if r['midi_exists'] else ("✗ ERROR" if r.get('error') else "- NO FILE")
            mi = r.get('mi_enso_pitch', 0.0)
            nmi = r.get('nmi_enso_pitch', 0.0)
            preserve = (mi / enso_entropy * 100) if enso_entropy > 0 else 0.0
            
            f.write(f"{r['mode']:<25} {status:<10} {mi:>8.4f}    {nmi:>6.4f}    {preserve:>8.2f}\n")
        
        f.write("\n\n")
        
        # Detailed results per mode
        f.write("="*80 + "\n")
        f.write("DETAILED ANALYSIS BY MODE\n")
        f.write("="*80 + "\n\n")
        
        for r in all_results:
            f.write(f"MODE: {r['mode'].upper()}\n")
            f.write("-"*80 + "\n\n")
            
            if r.get('error'):
                f.write(f"✗ ANALYSIS FAILED\n")
                f.write(f"Error: {r['error']}\n\n")
                continue
            
            if not r['midi_exists'] and not r['wav_exists']:
                f.write("- No music files found for this mode\n\n")
                continue
            
            if r['midi_exists']:
                f.write("MIDI ANALYSIS (Pitch Sequence)\n")
                f.write("."*40 + "\n")
                f.write(f"Shannon Entropy [bits]:               {r.get('pitch_entropy', 0):.4f} (bins={r.get('pitch_bins', 0)})\n")
                
                se = r.get('pitch_sample_entropy', np.nan)
                se_str = f"{se:.4f}" if not np.isnan(se) else "N/A"
                f.write(f"Sample Entropy [dimensionless]:       {se_str}\n")
                
                ae = r.get('pitch_approx_entropy', np.nan)
                ae_str = f"{ae:.4f}" if not np.isnan(ae) else "N/A"
                f.write(f"Approximate Entropy [dimensionless]:  {ae_str}\n")
                
                f.write(f"Permutation Entropy [normalized]:     {r.get('pitch_perm_entropy', 0):.4f}\n")
                f.write(f"Spectral Entropy [normalized]:        {r.get('pitch_spectral_entropy', 0):.4f}\n")
                f.write("\n")
                
                f.write("MIDI ANALYSIS (Velocity)\n")
                f.write("."*40 + "\n")
                f.write(f"Shannon Entropy [bits]:               {r.get('velocity_entropy', 0):.4f} (bins={r.get('velocity_bins', 0)})\n")
                f.write("\n")
                
                f.write("INFORMATION COUPLING WITH ENSO\n")
                f.write("."*40 + "\n")
                f.write(f"Mutual Information (ENSO,Pitch) [bits]:     {r.get('mi_enso_pitch', 0):.4f}\n")
                f.write(f"Normalized MI (ENSO,Pitch) [0-1]:           {r.get('nmi_enso_pitch', 0):.4f}\n")
                f.write(f"Mutual Information (ENSO,Velocity) [bits]:  {r.get('mi_enso_velocity', 0):.4f}\n")
                
                preservation = (r.get('mi_enso_pitch', 0) / enso_entropy * 100) if enso_entropy > 0 else 0
                f.write(f"Information Preservation [%]:                {preservation:.2f}\n")
                f.write("\n")
            
            if r['wav_exists']:
                f.write("AUDIO ANALYSIS (WAV Features)\n")
                f.write("."*40 + "\n")
                f.write(f"Spectral Centroid Entropy [bits]:     {r.get('spectral_centroid_entropy', 0):.4f}\n")
                f.write(f"Zero-Crossing Rate Entropy [bits]:    {r.get('zcr_entropy', 0):.4f}\n")
                f.write(f"MI (ENSO,Spectral Centroid) [bits]:   {r.get('mi_enso_spectral_centroid', 0):.4f}\n")
                f.write("\n")
            
            f.write("."*80 + "\n\n")
        
        f.write("="*80 + "\n")
        f.write("NOTES ON METRICS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Shannon Entropy [bits]:\n")
        f.write("  Measures information content. Higher values indicate more variability\n")
        f.write("  and unpredictability in the signal.\n\n")
        
        f.write("Sample Entropy [dimensionless]:\n")
        f.write("  Quantifies signal complexity and regularity. Higher values suggest\n")
        f.write("  more complex, less predictable patterns.\n\n")
        
        f.write("Approximate Entropy [dimensionless]:\n")
        f.write("  Similar to Sample Entropy but slightly different calculation.\n")
        f.write("  Measures pattern regularity in time series.\n\n")
        
        f.write("Permutation Entropy [0-1 normalized]:\n")
        f.write("  Captures ordinal patterns in time series. Value of 1 indicates\n")
        f.write("  maximum randomness; 0 indicates complete order.\n\n")
        
        f.write("Spectral Entropy [0-1 normalized]:\n")
        f.write("  Measures frequency domain complexity. Higher values indicate\n")
        f.write("  broader frequency distribution.\n\n")
        
        f.write("Mutual Information (MI) [bits]:\n")
        f.write("  Quantifies statistical dependence between ENSO and music signals.\n")
        f.write("  Higher values mean more shared information.\n\n")
        
        f.write("Normalized MI (NMI) [0-1]:\n")
        f.write("  Mutual Information normalized by average entropy of both signals.\n")
        f.write("  Value of 1 indicates perfect dependence; 0 indicates independence.\n\n")
        
        f.write("Information Preservation [%]:\n")
        f.write("  Percentage of ENSO's information content retained in music.\n")
        f.write("  Calculated as (MI / ENSO_entropy) × 100.\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF ANALYSIS\n")
        f.write("="*80 + "\n")


def write_results_csv(all_results, enso_data, output_file):
    """Write results to CSV file."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'mode',
            'status',
            'midi_exists',
            'wav_exists',
            'error',
            'pitch_entropy_bits',
            'pitch_bins',
            'velocity_entropy_bits',
            'velocity_bins',
            'pitch_sample_entropy_dimensionless',
            'pitch_approx_entropy_dimensionless',
            'pitch_perm_entropy_normalized',
            'pitch_spectral_entropy_normalized',
            'mi_enso_pitch_bits',
            'nmi_enso_pitch_normalized',
            'mi_enso_velocity_bits',
            'spectral_centroid_entropy_bits',
            'zcr_entropy_bits',
            'mi_enso_spectral_centroid_bits',
            'information_preservation_percent'
        ])
        
        enso_entropy, _ = shannon_entropy(enso_data)
        
        # Data rows
        for r in all_results:
            status = "SUCCESS" if r['midi_exists'] else ("ERROR" if r.get('error') else "NO_FILE")
            
            mi = r.get('mi_enso_pitch', 0.0)
            preserve = (mi / enso_entropy * 100) if enso_entropy > 0 else 0.0
            
            se = r.get('pitch_sample_entropy', np.nan)
            ae = r.get('pitch_approx_entropy', np.nan)
            
            writer.writerow([
                r['mode'],
                status,
                int(r['midi_exists']),
                int(r['wav_exists']),
                r.get('error', ''),
                f"{r.get('pitch_entropy', 0):.6f}",
                r.get('pitch_bins', 0),
                f"{r.get('velocity_entropy', 0):.6f}",
                r.get('velocity_bins', 0),
                f"{se:.6f}" if not np.isnan(se) else '',
                f"{ae:.6f}" if not np.isnan(ae) else '',
                f"{r.get('pitch_perm_entropy', 0):.6f}",
                f"{r.get('pitch_spectral_entropy', 0):.6f}",
                f"{r.get('mi_enso_pitch', 0):.6f}",
                f"{r.get('nmi_enso_pitch', 0):.6f}",
                f"{r.get('mi_enso_velocity', 0):.6f}",
                f"{r.get('spectral_centroid_entropy', 0):.6f}",
                f"{r.get('zcr_entropy', 0):.6f}",
                f"{r.get('mi_enso_spectral_centroid', 0):.6f}",
                f"{preserve:.2f}"
            ])


def main():
    print("\n" + "="*70)
    print("PARALLEL MUSIC ANALYSIS")
    print("="*70 + "\n")
    
    STATS_DIR.mkdir(exist_ok=True)
    
    print("Loading ENSO data...")
    enso_data = load_enso_data(INPUT_FILE)
    print(f"  Loaded {len(enso_data)} months of data\n")
    
    print("Analyzing sonification modes in parallel...")
    n_cores = min(cpu_count(), len(MODES))
    print(f"  Using {n_cores} CPU cores")
    
    # Prepare arguments for parallel processing
    args_list = [(mode, enso_data) for mode in MODES]
    
    # Process in parallel
    with Pool(n_cores) as pool:
        all_results = pool.map(analyze_single_mode, args_list)
    
    modes_found = sum([1 for r in all_results if r['midi_exists']])
    print(f"\n  Successfully analyzed {modes_found} modes")
    
    # Save results
    print("\nSaving results...")
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"  Saved: {OUTPUT_PKL.name}")
    
    # Print summary
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70 + "\n")
    
    for r in all_results:
        if r['midi_exists']:
            print(f"{r['mode']:<25}")
            print(f"  Pitch Entropy:    {r['pitch_entropy']:.4f} bits")
            print(f"  MI(ENSO,Pitch):   {r['mi_enso_pitch']:.4f} bits")
            print(f"  NMI:              {r['nmi_enso_pitch']:.4f}")
            print()
    
    print("="*70)
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
