#!/usr/bin/env python
"""
music_characterization_revised.py

Musical Characterization Analysis from Audio
==============================================================

Authors: Sandy H. S. Herho, Rusmawan Suwarman, Nurjanna J. Trilaksono
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
from scipy.fft import fft, fftfreq
import csv
from datetime import datetime

warnings.filterwarnings('ignore')

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

MUSIC_DIR = Path('../music_outputs')
STATS_DIR = Path('../stats')
OUTPUT_DIR = Path('../output_data')

MODES = ['pelog_layered', 'pelog_alternating', 'pelog_melodic', 'pelog_spectral',
         'slendro_layered', 'slendro_alternating', 'slendro_melodic', 'slendro_spectral']

# Improved parameters
HOP_LENGTH = 1024  # Increased from 512 for smoother trajectories
SMOOTHING_WINDOW = 20  # Increased from 10


def load_audio(audio_file, sr=22050):
    """Load audio file."""
    try:
        if LIBROSA_AVAILABLE:
            y, sr = librosa.load(audio_file, sr=sr, mono=True)
            return y, sr
        elif SOUNDFILE_AVAILABLE:
            y, sr = sf.read(audio_file)
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
            return y, sr
        else:
            return None, None
    except Exception as e:
        print(f"    Error loading {audio_file.name}: {e}")
        return None, None


def extract_spectral_centroid(y, sr, hop_length=HOP_LENGTH):
    """Extract spectral centroid."""
    if LIBROSA_AVAILABLE:
        return librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    else:
        frame_length = 2048
        n_frames = 1 + (len(y) - frame_length) // hop_length
        centroid = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * hop_length
            frame = y[start:start+frame_length]
            if len(frame) < frame_length:
                frame = np.pad(frame, (0, frame_length - len(frame)))
            window = np.hanning(len(frame))
            frame = frame * window
            spectrum = np.abs(fft(frame))[:len(frame)//2]
            freqs = fftfreq(len(frame), 1/sr)[:len(frame)//2]
            if np.sum(spectrum) > 0:
                centroid[i] = np.sum(freqs * spectrum) / np.sum(spectrum)
        return centroid


def extract_rms_energy(y, sr, hop_length=HOP_LENGTH, frame_length=2048):
    """Extract RMS energy."""
    if LIBROSA_AVAILABLE:
        return librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    else:
        n_frames = 1 + (len(y) - frame_length) // hop_length
        rms = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * hop_length
            frame = y[start:start+frame_length]
            if len(frame) > 0:
                rms[i] = np.sqrt(np.mean(frame**2))
        return rms


def extract_zero_crossing_rate(y, hop_length=HOP_LENGTH, frame_length=2048):
    """Extract zero crossing rate."""
    if LIBROSA_AVAILABLE:
        return librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    else:
        n_frames = 1 + (len(y) - frame_length) // hop_length
        zcr = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * hop_length
            frame = y[start:start+frame_length]
            if len(frame) > 1:
                zcr[i] = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
        return zcr


def extract_spectral_rolloff(y, sr, hop_length=HOP_LENGTH):
    """Extract spectral rolloff."""
    if LIBROSA_AVAILABLE:
        return librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    else:
        frame_length = 2048
        n_frames = 1 + (len(y) - frame_length) // hop_length
        rolloff = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * hop_length
            frame = y[start:start+frame_length]
            if len(frame) < frame_length:
                frame = np.pad(frame, (0, frame_length - len(frame)))
            window = np.hanning(len(frame))
            frame = frame * window
            spectrum = np.abs(fft(frame))[:len(frame)//2]
            cumsum = np.cumsum(spectrum)
            if cumsum[-1] > 0:
                threshold = 0.85 * cumsum[-1]
                idx = np.where(cumsum >= threshold)[0]
                if len(idx) > 0:
                    rolloff[i] = fftfreq(len(frame), 1/sr)[idx[0]]
        return rolloff


def extract_spectral_bandwidth(y, sr, hop_length=HOP_LENGTH):
    """Extract spectral bandwidth."""
    if LIBROSA_AVAILABLE:
        return librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    else:
        frame_length = 2048
        n_frames = 1 + (len(y) - frame_length) // hop_length
        bandwidth = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * hop_length
            frame = y[start:start+frame_length]
            if len(frame) < frame_length:
                frame = np.pad(frame, (0, frame_length - len(frame)))
            window = np.hanning(len(frame))
            frame = frame * window
            spectrum = np.abs(fft(frame))[:len(frame)//2]
            if np.sum(spectrum) > 0:
                bandwidth[i] = np.std(spectrum)
        return bandwidth


def calculate_brightness(spectral_centroid):
    """Average spectral centroid."""
    return np.mean(spectral_centroid)


def calculate_intensity(rms_energy):
    """Average RMS energy."""
    return np.mean(rms_energy)


def calculate_dynamic_range_audio(rms_energy):
    """Ratio of max to mean energy."""
    mean_energy = np.mean(rms_energy)
    max_energy = np.max(rms_energy)
    if mean_energy > 0:
        return max_energy / mean_energy
    return 1.0


def calculate_energy_variation(rms_energy):
    """Coefficient of variation in energy."""
    if len(rms_energy) > 0 and np.mean(rms_energy) > 0:
        return np.std(rms_energy) / np.mean(rms_energy)
    return 0


def calculate_brightness_stability(spectral_centroid):
    """Stability of brightness."""
    if len(spectral_centroid) > 0 and np.mean(spectral_centroid) > 0:
        return np.std(spectral_centroid) / np.mean(spectral_centroid)
    return 0


def calculate_percussiveness(zcr):
    """Average zero crossing rate."""
    return np.mean(zcr)


def calculate_high_frequency_content(spectral_rolloff, sr):
    """Average spectral rolloff."""
    return np.mean(spectral_rolloff) / (sr / 2)


def calculate_spectral_complexity(spectral_bandwidth):
    """Average spectral bandwidth."""
    return np.mean(spectral_bandwidth)


def calculate_temporal_regularity(rms_energy):
    """Regularity of energy envelope."""
    if len(rms_energy) < 20:
        return 0
    rms_norm = (rms_energy - np.mean(rms_energy)) / (np.std(rms_energy) + 1e-10)
    autocorr = np.correlate(rms_norm, rms_norm, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]
    if len(autocorr) > 10:
        peaks, _ = signal.find_peaks(autocorr[1:50])
        if len(peaks) > 0:
            return autocorr[peaks[0] + 1]
    return 0


def calculate_spectral_flux(y, sr, hop_length=HOP_LENGTH):
    """Calculate spectral flux."""
    if LIBROSA_AVAILABLE:
        S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length)
        S_db = librosa.power_to_db(S, ref=np.max)
        flux = np.sum(np.abs(np.diff(S_db, axis=1)), axis=0)
        return np.mean(flux)
    else:
        frame_length = 2048
        n_frames = 1 + (len(y) - frame_length) // hop_length
        flux_values = []
        prev_spectrum = None
        for i in range(n_frames):
            start = i * hop_length
            frame = y[start:start+frame_length]
            if len(frame) < frame_length:
                frame = np.pad(frame, (0, frame_length - len(frame)))
            window = np.hanning(len(frame))
            frame = frame * window
            spectrum = np.abs(fft(frame))[:len(frame)//2]
            if prev_spectrum is not None:
                flux_values.append(np.sum(np.abs(spectrum - prev_spectrum)))
            prev_spectrum = spectrum
        return np.mean(flux_values) if flux_values else 0


def extract_raw_features(audio_file, sr=22050, hop_length=HOP_LENGTH):
    """Extract raw features (no normalization yet)."""
    try:
        y, sr_actual = load_audio(audio_file, sr=sr)
        if y is None:
            return None
        
        spectral_centroid = extract_spectral_centroid(y, sr_actual, hop_length=hop_length)
        rms_energy = extract_rms_energy(y, sr_actual, hop_length=hop_length)
        
        n_frames = len(spectral_centroid)
        times = librosa.frames_to_time(np.arange(n_frames), sr=sr_actual, hop_length=hop_length) if LIBROSA_AVAILABLE else np.arange(n_frames) * hop_length / sr_actual
        
        df = pd.DataFrame({
            'time_step': np.arange(n_frames),
            'time_sec': times,
            'spectral_centroid': spectral_centroid,
            'rms_energy': rms_energy
        })
        
        return df
    except Exception as e:
        print(f"    Error: {e}")
        return None


def analyze_single_mode(mode_name):
    """Analyze one sonification mode from audio."""
    print(f"  Analyzing: {mode_name}")
    
    result = {'mode': mode_name, 'success': False}
    
    try:
        audio_file = MUSIC_DIR / f'enso_javanese_{mode_name}.wav'
        if not audio_file.exists():
            result['error'] = 'WAV file not found'
            return result
        
        y, sr = load_audio(audio_file, sr=22050)
        if y is None:
            result['error'] = 'Failed to load audio'
            return result
        
        spectral_centroid = extract_spectral_centroid(y, sr, hop_length=HOP_LENGTH)
        rms_energy = extract_rms_energy(y, sr, hop_length=HOP_LENGTH)
        zcr = extract_zero_crossing_rate(y, hop_length=HOP_LENGTH)
        spectral_rolloff = extract_spectral_rolloff(y, sr, hop_length=HOP_LENGTH)
        spectral_bandwidth = extract_spectral_bandwidth(y, sr, hop_length=HOP_LENGTH)
        
        result['brightness'] = calculate_brightness(spectral_centroid)
        result['intensity'] = calculate_intensity(rms_energy)
        result['dynamic_range'] = calculate_dynamic_range_audio(rms_energy)
        result['energy_variation'] = calculate_energy_variation(rms_energy)
        result['brightness_stability'] = calculate_brightness_stability(spectral_centroid)
        result['percussiveness'] = calculate_percussiveness(zcr)
        result['high_freq_content'] = calculate_high_frequency_content(spectral_rolloff, sr)
        result['spectral_complexity'] = calculate_spectral_complexity(spectral_bandwidth)
        result['temporal_regularity'] = calculate_temporal_regularity(rms_energy)
        result['spectral_flux'] = calculate_spectral_flux(y, sr, hop_length=HOP_LENGTH)
        
        result['duration_sec'] = len(y) / sr
        result['sample_rate'] = sr
        result['num_samples'] = len(y)
        result['avg_spectral_centroid'] = np.mean(spectral_centroid)
        result['avg_rms_energy'] = np.mean(rms_energy)
        
        result['success'] = True
    except Exception as e:
        result['error'] = str(e)
        print(f"    Error: {e}")
    
    return result


def compare_modes(all_results):
    """Calculate comparative rankings."""
    modes = [r for r in all_results if r['success']]
    if len(modes) == 0:
        return all_results
    
    metrics = ['brightness', 'intensity', 'dynamic_range', 'energy_variation', 
               'brightness_stability', 'percussiveness', 'high_freq_content', 
               'spectral_complexity', 'temporal_regularity', 'spectral_flux']
    
    for metric in metrics:
        values = [m[metric] for m in modes]
        ranks = stats.rankdata(values, method='average')
        for i, mode in enumerate(modes):
            mode[f'{metric}_rank'] = int(ranks[i])
    
    for mode in modes:
        if 'pelog' in mode['mode']:
            mode['scale_family'] = 'Pelog'
        else:
            mode['scale_family'] = 'Slendro'
        
        if 'layered' in mode['mode']:
            mode['composition_type'] = 'Layered'
        elif 'alternating' in mode['mode']:
            mode['composition_type'] = 'Alternating'
        elif 'melodic' in mode['mode']:
            mode['composition_type'] = 'Melodic'
        else:
            mode['composition_type'] = 'Spectral'
    
    return all_results


def write_characterization_report(all_results, output_file):
    """Write detailed report."""
    modes = [r for r in all_results if r['success']]
    
    with open(output_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("GAMELAN SONIFICATION MUSICAL CHARACTERIZATION (REVISED)\n")
        f.write("="*100 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Improvements: Global normalization, increased smoothing (window=20), hop_length=1024\n")
        f.write("="*100 + "\n\n")
        
        f.write("SUMMARY TABLE\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Mode':<25} {'Bright(Hz)':<12} {'Intens':<10} {'DynRng':<8} {'Percuss':<8} {'Flux':<8}\n")
        f.write("-"*100 + "\n")
        
        for r in modes:
            f.write(f"{r['mode']:<25} {r['brightness']:>10.1f}  ")
            f.write(f"{r['intensity']:>8.4f}  ")
            f.write(f"{r['dynamic_range']:>6.3f}  ")
            f.write(f"{r['percussiveness']:>6.3f}  ")
            f.write(f"{r['spectral_flux']:>6.1f}\n")
        
        f.write("\n")
        f.write("="*100 + "\n")


def write_csv_results(all_results, output_file):
    """Write results to CSV."""
    modes = [r for r in all_results if r['success']]
    if len(modes) == 0:
        return
    
    fieldnames = list(modes[0].keys())
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(modes)


def compute_global_normalization_params(all_modes):
    """
    Compute global normalization parameters across all modes.
    This ensures all modes are on the same scale.
    """
    print("\n  Computing global normalization parameters...")
    
    all_centroids = []
    all_energies = []
    
    for mode in all_modes:
        audio_file = MUSIC_DIR / f'enso_javanese_{mode}.wav'
        if not audio_file.exists():
            continue
        
        df = extract_raw_features(audio_file, sr=22050, hop_length=HOP_LENGTH)
        if df is not None:
            all_centroids.extend(df['spectral_centroid'].values)
            all_energies.extend(df['rms_energy'].values)
    
    if len(all_centroids) == 0:
        return None
    
    # Use percentiles for robustness to outliers
    centroid_min = np.percentile(all_centroids, 1)
    centroid_max = np.percentile(all_centroids, 99)
    
    energy_max = np.percentile(all_energies, 99)
    
    params = {
        'centroid_min': centroid_min,
        'centroid_max': centroid_max,
        'centroid_range': centroid_max - centroid_min,
        'energy_max': energy_max
    }
    
    print(f"    Spectral Centroid: {centroid_min:.1f} - {centroid_max:.1f} Hz")
    print(f"    RMS Energy Max: {energy_max:.6f}")
    
    return params


def write_state_variables_with_global_norm(all_results, output_dir, norm_params):
    """
    Write dynamic state variable time series with GLOBAL normalization.
    """
    modes = [r for r in all_results if r['success']]
    if len(modes) == 0 or norm_params is None:
        return 0
    
    extracted_count = 0
    print("\n  Extracting state variables with global normalization...")
    
    for r in modes:
        mode_name = r['mode']
        audio_file = MUSIC_DIR / f'enso_javanese_{mode_name}.wav'
        
        if not audio_file.exists():
            continue
        
        print(f"    {mode_name}")
        
        # Extract raw features
        df = extract_raw_features(audio_file, sr=22050, hop_length=HOP_LENGTH)
        
        if df is None:
            print(f"      ✗ Failed")
            continue
        
        # Apply GLOBAL normalization
        df['spectral_centroid_norm'] = (df['spectral_centroid'] - norm_params['centroid_min']) / norm_params['centroid_range']
        df['spectral_centroid_norm'] = df['spectral_centroid_norm'].clip(0, 1)
        
        df['rms_energy_norm'] = df['rms_energy'] / norm_params['energy_max']
        df['rms_energy_norm'] = df['rms_energy_norm'].clip(0, 1)
        
        # Apply INCREASED smoothing (window=20)
        df['spectral_centroid_smooth'] = df['spectral_centroid'].rolling(
            window=SMOOTHING_WINDOW, center=True, min_periods=1
        ).mean()
        
        df['rms_energy_smooth'] = df['rms_energy'].rolling(
            window=SMOOTHING_WINDOW, center=True, min_periods=1
        ).mean()
        
        # Normalized smoothed versions with GLOBAL scale
        df['spectral_centroid_smooth_norm'] = (df['spectral_centroid_smooth'] - norm_params['centroid_min']) / norm_params['centroid_range']
        df['spectral_centroid_smooth_norm'] = df['spectral_centroid_smooth_norm'].clip(0, 1)
        
        df['rms_energy_smooth_norm'] = df['rms_energy_smooth'] / norm_params['energy_max']
        df['rms_energy_smooth_norm'] = df['rms_energy_smooth_norm'].clip(0, 1)
        
        # Add metadata
        df['mode'] = mode_name
        df['scale_family'] = r['scale_family']
        df['composition_type'] = r['composition_type']
        
        # Reorder columns
        df = df[['mode', 'scale_family', 'composition_type', 'time_step', 'time_sec',
                 'spectral_centroid', 'rms_energy', 'spectral_centroid_norm', 'rms_energy_norm',
                 'spectral_centroid_smooth', 'rms_energy_smooth',
                 'spectral_centroid_smooth_norm', 'rms_energy_smooth_norm']]
        
        # Save
        output_file = output_dir / f'state_variables_{mode_name}.csv'
        df.to_csv(output_file, index=False, float_format='%.6f')
        print(f"      ✓ {output_file.name} ({len(df)} time steps)")
        extracted_count += 1
    
    # Write README
    readme_file = output_dir / 'STATE_VARIABLES_README.txt'
    with open(readme_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DYNAMIC MUSICAL STATE VARIABLES WITH GLOBAL NORMALIZATION\n")
        f.write("="*80 + "\n\n")
        f.write(f"NORMALIZATION PARAMETERS:\n")
        f.write(f"  Spectral Centroid: {norm_params['centroid_min']:.1f} - {norm_params['centroid_max']:.1f} Hz\n")
        f.write(f"  RMS Energy: 0 - {norm_params['energy_max']:.6f}\n\n")
        f.write("USE THESE COLUMNS FOR PHASE SPACE PLOTS:\n")
        f.write("  - spectral_centroid_smooth_norm (0-1 scale, globally normalized)\n")
        f.write("  - rms_energy_smooth_norm (0-1 scale, globally normalized)\n\n")
        f.write("="*80 + "\n")
    
    print(f"    ✓ STATE_VARIABLES_README.txt")
    
    return extracted_count


def main():
    print("\n" + "="*70)
    print("MUSICAL CHARACTERIZATION ANALYSIS - REVISED VERSION")
    print("="*70)
    print("\nImprovements:")
    print("  ✓ Global normalization across all modes")
    print("  ✓ Increased smoothing window (20 frames)")
    print("  ✓ Better hop length (1024 samples)")
    print("  ✓ Percentile-based robust normalization\n")
    
    if not LIBROSA_AVAILABLE and not SOUNDFILE_AVAILABLE:
        print("ERROR: No audio library available!")
        print("Please install: pip install librosa")
        return
    
    STATS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Step 1: Compute global normalization parameters
    norm_params = compute_global_normalization_params(MODES)
    
    if norm_params is None:
        print("ERROR: Could not compute normalization parameters")
        return
    
    # Step 2: Analyze modes
    print("\nAnalyzing musical characteristics...\n")
    
    all_results = []
    for mode in MODES:
        result = analyze_single_mode(mode)
        all_results.append(result)
    
    all_results = compare_modes(all_results)
    
    success_count = sum([1 for r in all_results if r['success']])
    print(f"\nSuccessfully analyzed {success_count}/{len(MODES)} modes\n")
    
    if success_count == 0:
        print("No modes analyzed successfully.")
        return
    
    # Step 3: Write results
    print("Writing results...")
    
    report_file = STATS_DIR / 'music_characterization_revised.txt'
    write_characterization_report(all_results, report_file)
    print(f"  ✓ {report_file.name}")
    
    csv_file = STATS_DIR / 'music_characterization_revised.csv'
    write_csv_results(all_results, csv_file)
    print(f"  ✓ {csv_file.name}")
    
    # Step 4: Write state variables with global normalization
    state_count = write_state_variables_with_global_norm(all_results, OUTPUT_DIR, norm_params)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nState variables: {OUTPUT_DIR}/ ({state_count} time series)")
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
