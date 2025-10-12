#!/usr/bin/env python
"""
music_results_extractor.py

Extract and Report Music Analysis Results
==========================================

Comprehensive extraction of music analysis results from pickle file to:
- Detailed CSV files (summary + per-mode)
- Full descriptive statistics with entropy calculations
- Information-theoretic analysis report

Authors: Sandy H. S. Herho, Rusmawan Suwarman, Edi Riawan, Nurjanna J. Trilaksono
Institution: Weather and Climate Prediction Laboratory (WCPL) ITB
Date: 10/12/2025
License: WTFPL
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import csv
from datetime import datetime

# Configuration
INPUT_DIR = Path('../input_data')
STATS_DIR = Path('../stats')
INPUT_FILE = INPUT_DIR / 'nino34_hadisst_mon_1870_2024.csv'
PICKLE_FILE = STATS_DIR / 'music_analysis_results.pkl'
OUTPUT_SUMMARY_CSV = STATS_DIR / 'music_summary.csv'
OUTPUT_DETAILED_CSV = STATS_DIR / 'music_detailed.csv'
OUTPUT_ENTROPY_TXT = STATS_DIR / 'music_entropy_full_calculations.txt'
OUTPUT_STATISTICS_TXT = STATS_DIR / 'music_descriptive_statistics.txt'


def load_enso_data(filepath):
    """Load ENSO Nino 3.4 data."""
    df = pd.read_csv(filepath, skipinitialspace=True)
    df.columns = df.columns.str.strip().str.lower()
    df['date'] = pd.to_datetime(df['date'])
    df['nino34'] = pd.to_numeric(df['nino34'], errors='coerce')
    df.loc[df['nino34'] < -900, 'nino34'] = np.nan
    df = df.dropna(subset=['nino34'])
    return df['nino34'].values


def shannon_entropy(data):
    """Calculate Shannon entropy with binning details."""
    if len(data) < 2:
        return 0.0, 5, {}
    
    data_clean = data[np.isfinite(data)]
    if len(data_clean) < 2:
        return 0.0, 5, {}
    
    # Calculate bins using Freedman-Diaconis rule
    iqr = np.percentile(data_clean, 75) - np.percentile(data_clean, 25)
    if iqr == 0:
        bins = int(np.ceil(np.log2(len(data_clean)) + 1))
    else:
        h = 2 * iqr / (len(data_clean)**(1/3))
        bins = int(np.ceil((np.max(data_clean) - np.min(data_clean)) / h)) if h > 0 else 30
    
    bins = max(5, min(bins, 100))
    
    try:
        hist, bin_edges = np.histogram(data_clean, bins=bins, density=True)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0, bins, {}
        
        bin_width = (np.max(data_clean) - np.min(data_clean)) / bins
        prob = hist * bin_width
        prob = prob / np.sum(prob)
        
        entropy = -np.sum(prob * np.log2(prob + 1e-10))
        
        details = {
            'n_samples': len(data_clean),
            'n_bins': bins,
            'bin_width': bin_width,
            'min_val': np.min(data_clean),
            'max_val': np.max(data_clean),
            'range': np.max(data_clean) - np.min(data_clean),
            'iqr': iqr,
            'mean': np.mean(data_clean),
            'std': np.std(data_clean),
            'entropy_bits': entropy,
            'max_entropy': np.log2(bins),
            'efficiency': entropy / np.log2(bins) if bins > 1 else 0
        }
        
        return entropy, bins, details
    except:
        return 0.0, bins, {}


def create_summary_csv(all_results, enso_data, output_file):
    """Create summary CSV with key metrics."""
    print("  Creating summary CSV...")
    
    enso_entropy, _, _ = shannon_entropy(enso_data)
    
    summary_data = []
    
    for r in all_results:
        if not r['midi_exists']:
            continue
        
        scale = 'Pelog' if 'pelog' in r['mode'] else 'Slendro'
        composition = r['mode'].split('_')[1].capitalize()
        
        mi_pitch = r.get('mi_enso_pitch', 0.0)
        preservation = (mi_pitch / enso_entropy * 100) if enso_entropy > 0 else 0.0
        
        se = r.get('pitch_sample_entropy', np.nan)
        ae = r.get('pitch_approx_entropy', np.nan)
        
        summary_data.append({
            'Mode': r['mode'],
            'Scale': scale,
            'Composition': composition,
            'Status': 'Success' if r['midi_exists'] else 'Failed',
            'Pitch_Entropy_bits': f"{r.get('pitch_entropy', 0):.6f}",
            'Velocity_Entropy_bits': f"{r.get('velocity_entropy', 0):.6f}",
            'Sample_Entropy': f"{se:.6f}" if not np.isnan(se) else 'N/A',
            'Approx_Entropy': f"{ae:.6f}" if not np.isnan(ae) else 'N/A',
            'Perm_Entropy': f"{r.get('pitch_perm_entropy', 0):.6f}",
            'Spectral_Entropy': f"{r.get('pitch_spectral_entropy', 0):.6f}",
            'MI_ENSO_Pitch_bits': f"{mi_pitch:.6f}",
            'NMI_ENSO_Pitch': f"{r.get('nmi_enso_pitch', 0):.6f}",
            'MI_ENSO_Velocity_bits': f"{r.get('mi_enso_velocity', 0):.6f}",
            'Info_Preservation_percent': f"{preservation:.2f}"
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(output_file, index=False)
    print(f"    Saved: {output_file.name} ({len(df)} modes)")


def create_detailed_csv(all_results, output_file):
    """Create detailed CSV with all metrics and metadata."""
    print("  Creating detailed CSV...")
    
    detailed_data = []
    
    for r in all_results:
        scale = 'Pelog' if 'pelog' in r['mode'] else 'Slendro'
        composition = r['mode'].split('_')[1].capitalize()
        
        se = r.get('pitch_sample_entropy', np.nan)
        ae = r.get('pitch_approx_entropy', np.nan)
        
        detailed_data.append({
            'mode': r['mode'],
            'scale': scale,
            'composition_type': composition,
            'midi_exists': r['midi_exists'],
            'wav_exists': r['wav_exists'],
            'analysis_error': r.get('error', ''),
            
            # Pitch metrics
            'pitch_shannon_entropy_bits': r.get('pitch_entropy', 0),
            'pitch_bins_used': r.get('pitch_bins', 0),
            'pitch_sample_entropy': se if not np.isnan(se) else None,
            'pitch_approx_entropy': ae if not np.isnan(ae) else None,
            'pitch_perm_entropy_normalized': r.get('pitch_perm_entropy', 0),
            'pitch_spectral_entropy_normalized': r.get('pitch_spectral_entropy', 0),
            
            # Velocity metrics
            'velocity_shannon_entropy_bits': r.get('velocity_entropy', 0),
            'velocity_bins_used': r.get('velocity_bins', 0),
            
            # Information coupling with ENSO
            'mi_enso_pitch_bits': r.get('mi_enso_pitch', 0),
            'nmi_enso_pitch_normalized': r.get('nmi_enso_pitch', 0),
            'mi_enso_velocity_bits': r.get('mi_enso_velocity', 0),
            
            # Audio features (if available)
            'spectral_centroid_entropy_bits': r.get('spectral_centroid_entropy', 0) if r['wav_exists'] else None,
            'zcr_entropy_bits': r.get('zcr_entropy', 0) if r['wav_exists'] else None,
            'mi_enso_spectral_centroid_bits': r.get('mi_enso_spectral_centroid', 0) if r['wav_exists'] else None,
        })
    
    df = pd.DataFrame(detailed_data)
    df.to_csv(output_file, index=False)
    print(f"    Saved: {output_file.name} ({len(df)} modes)")


def write_entropy_calculations(all_results, enso_data, output_file):
    """Write full entropy calculations with mathematical details."""
    print("  Writing entropy calculations...")
    
    with open(output_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("FULL ENTROPY CALCULATIONS - MATHEMATICAL DETAILS\n")
        f.write("="*100 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Authors: Sandy H. S. Herho, Rusmawan Suwarman, Edi Riawan, Nurjanna J. Trilaksono\n")
        f.write("Institution: Weather and Climate Prediction Laboratory (WCPL) ITB\n")
        f.write("="*100 + "\n\n")
        
        # ENSO baseline entropy calculation
        f.write("BASELINE: ENSO NINO 3.4 INDEX\n")
        f.write("-"*100 + "\n\n")
        
        enso_entropy, enso_bins, enso_details = shannon_entropy(enso_data)
        
        f.write("Shannon Entropy Calculation:\n")
        f.write(f"  Number of samples (N):           {enso_details.get('n_samples', 0)}\n")
        f.write(f"  Data range:                      [{enso_details.get('min_val', 0):.4f}, {enso_details.get('max_val', 0):.4f}]\n")
        f.write(f"  Total range:                     {enso_details.get('range', 0):.4f}\n")
        f.write(f"  Mean:                            {enso_details.get('mean', 0):.4f}\n")
        f.write(f"  Std. Deviation:                  {enso_details.get('std', 0):.4f}\n")
        f.write(f"  Interquartile Range (IQR):       {enso_details.get('iqr', 0):.4f}\n")
        f.write(f"  Number of bins (Freedman-Diaconis): {enso_details.get('n_bins', 0)}\n")
        f.write(f"  Bin width:                       {enso_details.get('bin_width', 0):.6f}\n")
        f.write(f"\n")
        f.write(f"  Shannon Entropy H(X):            {enso_details.get('entropy_bits', 0):.6f} bits\n")
        f.write(f"  Maximum Entropy (uniform):       {enso_details.get('max_entropy', 0):.6f} bits\n")
        f.write(f"  Efficiency (H/H_max):            {enso_details.get('efficiency', 0):.4f}\n")
        f.write(f"\n")
        f.write(f"  Formula: H(X) = -∑ p(x_i) × log₂(p(x_i))\n")
        f.write(f"  where p(x_i) is the probability density at bin i\n")
        f.write("\n\n")
        
        # Sample and Approximate Entropy for ENSO
        from scipy import stats as sp_stats
        
        def sample_entropy_calc(data, m=2, r=None):
            if r is None:
                r = 0.2 * np.std(data)
            N = len(data)
            if N < 10:
                return np.nan, {}
            
            def _maxdist(x_i, x_j):
                return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
            
            def _phi(m):
                x = [[data[j] for j in range(i, i + m)] for i in range(N - m + 1)]
                C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) - 1 for x_i in x]
                return sum(C), N - m + 1
            
            try:
                phi_m_count, phi_m_total = _phi(m)
                phi_m_plus_1_count, phi_m_plus_1_total = _phi(m + 1)
                
                phi_m = phi_m_count / phi_m_total
                phi_m_plus_1 = phi_m_plus_1_count / phi_m_plus_1_total
                
                if phi_m == 0 or phi_m_plus_1 == 0:
                    return np.nan, {}
                
                se = -np.log(phi_m_plus_1 / phi_m)
                
                details = {
                    'm': m,
                    'r': r,
                    'r_fraction': r / np.std(data),
                    'N': N,
                    'phi_m': phi_m,
                    'phi_m_plus_1': phi_m_plus_1,
                    'sample_entropy': se
                }
                
                return se, details
            except:
                return np.nan, {}
        
        enso_se, enso_se_details = sample_entropy_calc(enso_data, m=2)
        
        f.write("Sample Entropy Calculation:\n")
        f.write(f"  Embedding dimension (m):         {enso_se_details.get('m', 2)}\n")
        f.write(f"  Tolerance (r):                   {enso_se_details.get('r', 0):.6f}\n")
        f.write(f"  r as fraction of σ:              {enso_se_details.get('r_fraction', 0):.4f}\n")
        f.write(f"  Number of samples (N):           {enso_se_details.get('N', 0)}\n")
        f.write(f"  Φ(m):                           {enso_se_details.get('phi_m', 0):.8f}\n")
        f.write(f"  Φ(m+1):                         {enso_se_details.get('phi_m_plus_1', 0):.8f}\n")
        f.write(f"\n")
        f.write(f"  Sample Entropy (SampEn):         {enso_se:.6f}\n")
        f.write(f"\n")
        f.write(f"  Formula: SampEn(m,r,N) = -ln[Φ(m+1) / Φ(m)]\n")
        f.write(f"  where Φ(m) is the frequency of template matches\n")
        f.write("\n\n")
        
        # Individual mode calculations
        f.write("="*100 + "\n")
        f.write("MODE-BY-MODE ENTROPY CALCULATIONS\n")
        f.write("="*100 + "\n\n")
        
        for r in all_results:
            if not r['midi_exists']:
                continue
            
            f.write(f"MODE: {r['mode'].upper()}\n")
            f.write("-"*100 + "\n\n")
            
            # We don't have the raw pitch data here, so we report the stored values
            f.write("PITCH SEQUENCE ENTROPY ANALYSIS\n")
            f.write("."*50 + "\n\n")
            
            f.write("1. Shannon Entropy (Discrete Distribution)\n")
            f.write(f"   H(Pitch) = {r.get('pitch_entropy', 0):.6f} bits\n")
            f.write(f"   Number of bins used: {r.get('pitch_bins', 0)}\n")
            f.write(f"   Interpretation: On average, each pitch value provides\n")
            f.write(f"                   {r.get('pitch_entropy', 0):.4f} bits of information\n\n")
            
            se = r.get('pitch_sample_entropy', np.nan)
            if not np.isnan(se):
                f.write("2. Sample Entropy (Regularity)\n")
                f.write(f"   SampEn(Pitch) = {se:.6f}\n")
                f.write(f"   Interpretation: {'Low regularity (complex)' if se > 1.0 else 'High regularity (simple)'}\n\n")
            
            ae = r.get('pitch_approx_entropy', np.nan)
            if not np.isnan(ae):
                f.write("3. Approximate Entropy (Pattern Complexity)\n")
                f.write(f"   ApEn(Pitch) = {ae:.6f}\n")
                f.write(f"   Interpretation: {'High complexity' if ae > 1.0 else 'Low complexity'}\n\n")
            
            pe = r.get('pitch_perm_entropy', 0)
            f.write("4. Permutation Entropy (Ordinal Patterns)\n")
            f.write(f"   PE(Pitch) = {pe:.6f} (normalized, 0-1 scale)\n")
            f.write(f"   Interpretation: {'Highly random' if pe > 0.8 else 'Moderately ordered' if pe > 0.5 else 'Highly ordered'}\n\n")
            
            spec_e = r.get('pitch_spectral_entropy', 0)
            f.write("5. Spectral Entropy (Frequency Domain)\n")
            f.write(f"   SE(Pitch) = {spec_e:.6f} (normalized, 0-1 scale)\n")
            f.write(f"   Interpretation: {'Broad spectrum' if spec_e > 0.8 else 'Narrow spectrum'}\n\n")
            
            f.write("\nVELOCITY SEQUENCE ENTROPY ANALYSIS\n")
            f.write("."*50 + "\n\n")
            
            f.write("Shannon Entropy (Discrete Distribution)\n")
            f.write(f"   H(Velocity) = {r.get('velocity_entropy', 0):.6f} bits\n")
            f.write(f"   Number of bins used: {r.get('velocity_bins', 0)}\n\n")
            
            f.write("\nINFORMATION-THEORETIC COUPLING WITH ENSO\n")
            f.write("."*50 + "\n\n")
            
            mi_pitch = r.get('mi_enso_pitch', 0)
            nmi_pitch = r.get('nmi_enso_pitch', 0)
            mi_velocity = r.get('mi_enso_velocity', 0)
            
            f.write("Mutual Information Analysis:\n\n")
            
            f.write("1. MI(ENSO, Pitch)\n")
            f.write(f"   I(X;Y) = {mi_pitch:.6f} bits\n")
            f.write(f"   Formula: I(X;Y) = ∑∑ p(x,y) × log₂[p(x,y)/(p(x)p(y))]\n")
            f.write(f"   Interpretation: {mi_pitch:.4f} bits of information are\n")
            f.write(f"                   shared between ENSO and pitch sequence\n\n")
            
            f.write("2. Normalized MI(ENSO, Pitch)\n")
            f.write(f"   NMI(X;Y) = {nmi_pitch:.6f}\n")
            f.write(f"   Formula: NMI = 2×I(X;Y) / [H(X) + H(Y)]\n")
            f.write(f"   Interpretation: {nmi_pitch*100:.2f}% normalized information overlap\n\n")
            
            f.write("3. MI(ENSO, Velocity)\n")
            f.write(f"   I(X;Y) = {mi_velocity:.6f} bits\n\n")
            
            preservation = (mi_pitch / enso_entropy * 100) if enso_entropy > 0 else 0
            f.write("Information Preservation:\n")
            f.write(f"   Preservation = MI(ENSO,Pitch) / H(ENSO) × 100%\n")
            f.write(f"                = {mi_pitch:.6f} / {enso_entropy:.6f} × 100%\n")
            f.write(f"                = {preservation:.2f}%\n")
            f.write(f"   Interpretation: This sonification preserves {preservation:.2f}%\n")
            f.write(f"                   of the original ENSO signal's information\n\n")
            
            if r['wav_exists']:
                f.write("\nAUDIO FEATURES ENTROPY (WAV Analysis)\n")
                f.write("."*50 + "\n\n")
                f.write(f"Spectral Centroid Entropy:  {r.get('spectral_centroid_entropy', 0):.6f} bits\n")
                f.write(f"Zero-Crossing Rate Entropy: {r.get('zcr_entropy', 0):.6f} bits\n")
                f.write(f"MI(ENSO, Spectral Centroid): {r.get('mi_enso_spectral_centroid', 0):.6f} bits\n\n")
            
            f.write("="*100 + "\n\n")
        
        f.write("\nGLOSSARY OF ENTROPY MEASURES\n")
        f.write("="*100 + "\n\n")
        
        f.write("Shannon Entropy [bits]:\n")
        f.write("  H(X) = -∑ p(x) log₂(p(x))\n")
        f.write("  Measures average information content. Higher = more unpredictable.\n")
        f.write("  Range: 0 to log₂(N) where N is number of unique values.\n\n")
        
        f.write("Sample Entropy [dimensionless]:\n")
        f.write("  SampEn(m,r,N) = -ln[A/B]\n")
        f.write("  where A = number of template matches for m+1 points\n")
        f.write("        B = number of template matches for m points\n")
        f.write("  Measures regularity. Higher = more complex/irregular.\n")
        f.write("  Typical range: 0 to 2+\n\n")
        
        f.write("Approximate Entropy [dimensionless]:\n")
        f.write("  ApEn(m,r,N) = Φ(m) - Φ(m+1)\n")
        f.write("  Similar to Sample Entropy, measures pattern regularity.\n")
        f.write("  Higher values = less predictable patterns.\n\n")
        
        f.write("Permutation Entropy [normalized 0-1]:\n")
        f.write("  PE = H(ordinal patterns) / log₂(m!)\n")
        f.write("  Based on ordinal patterns in time series.\n")
        f.write("  0 = completely ordered, 1 = completely random.\n\n")
        
        f.write("Spectral Entropy [normalized 0-1]:\n")
        f.write("  SE = -∑ p(f) log₂(p(f)) / log₂(N)\n")
        f.write("  where p(f) is normalized power at frequency f\n")
        f.write("  Measures frequency distribution. 1 = flat spectrum, 0 = single frequency.\n\n")
        
        f.write("Mutual Information [bits]:\n")
        f.write("  I(X;Y) = ∑∑ p(x,y) log₂[p(x,y)/(p(x)p(y))]\n")
        f.write("  Measures statistical dependence between two variables.\n")
        f.write("  0 = independent, higher = more dependent.\n\n")
        
        f.write("="*100 + "\n")
        f.write("END OF ENTROPY CALCULATIONS\n")
        f.write("="*100 + "\n")
    
    print(f"    Saved: {output_file.name}")


def write_descriptive_statistics(all_results, enso_data, output_file):
    """Write comprehensive descriptive statistics."""
    print("  Writing descriptive statistics...")
    
    with open(output_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("MUSIC ANALYSIS - COMPREHENSIVE DESCRIPTIVE STATISTICS\n")
        f.write("="*100 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Authors: Sandy H. S. Herho, Rusmawan Suwarman, Edi Riawan, Nurjanna J. Trilaksono\n")
        f.write("Institution: Weather and Climate Prediction Laboratory (WCPL) ITB\n")
        f.write("="*100 + "\n\n")
        
        # Overall statistics
        modes_with_data = [r for r in all_results if r['midi_exists']]
        total_modes = len(all_results)
        successful_modes = len(modes_with_data)
        
        f.write("DATASET OVERVIEW\n")
        f.write("-"*100 + "\n")
        f.write(f"Total modes analyzed:              {total_modes}\n")
        f.write(f"Successfully analyzed:             {successful_modes}\n")
        f.write(f"Failed analyses:                   {total_modes - successful_modes}\n")
        f.write(f"Modes with WAV files:              {sum([1 for r in all_results if r['wav_exists']])}\n")
        f.write("\n\n")
        
        # Entropy statistics by category
        f.write("ENTROPY STATISTICS ACROSS ALL MODES\n")
        f.write("-"*100 + "\n\n")
        
        pitch_entropies = [r['pitch_entropy'] for r in modes_with_data]
        velocity_entropies = [r['velocity_entropy'] for r in modes_with_data]
        
        f.write("Shannon Entropy - Pitch [bits]\n")
        f.write(f"  Mean:       {np.mean(pitch_entropies):>10.4f}\n")
        f.write(f"  Median:     {np.median(pitch_entropies):>10.4f}\n")
        f.write(f"  Std Dev:    {np.std(pitch_entropies):>10.4f}\n")
        f.write(f"  Min:        {np.min(pitch_entropies):>10.4f}\n")
        f.write(f"  Max:        {np.max(pitch_entropies):>10.4f}\n")
        f.write(f"  Range:      {np.max(pitch_entropies) - np.min(pitch_entropies):>10.4f}\n\n")
        
        f.write("Shannon Entropy - Velocity [bits]\n")
        f.write(f"  Mean:       {np.mean(velocity_entropies):>10.4f}\n")
        f.write(f"  Median:     {np.median(velocity_entropies):>10.4f}\n")
        f.write(f"  Std Dev:    {np.std(velocity_entropies):>10.4f}\n")
        f.write(f"  Min:        {np.min(velocity_entropies):>10.4f}\n")
        f.write(f"  Max:        {np.max(velocity_entropies):>10.4f}\n")
        f.write(f"  Range:      {np.max(velocity_entropies) - np.min(velocity_entropies):>10.4f}\n\n")
        
        # Sample and Approximate Entropy
        sample_entropies = [r.get('pitch_sample_entropy', np.nan) for r in modes_with_data]
        sample_entropies_clean = [x for x in sample_entropies if not np.isnan(x)]
        
        if len(sample_entropies_clean) > 0:
            f.write("Sample Entropy [dimensionless]\n")
            f.write(f"  Mean:       {np.mean(sample_entropies_clean):>10.4f}\n")
            f.write(f"  Median:     {np.median(sample_entropies_clean):>10.4f}\n")
            f.write(f"  Std Dev:    {np.std(sample_entropies_clean):>10.4f}\n")
            f.write(f"  Min:        {np.min(sample_entropies_clean):>10.4f}\n")
            f.write(f"  Max:        {np.max(sample_entropies_clean):>10.4f}\n\n")
        
        approx_entropies = [r.get('pitch_approx_entropy', np.nan) for r in modes_with_data]
        approx_entropies_clean = [x for x in approx_entropies if not np.isnan(x)]
        
        if len(approx_entropies_clean) > 0:
            f.write("Approximate Entropy [dimensionless]\n")
            f.write(f"  Mean:       {np.mean(approx_entropies_clean):>10.4f}\n")
            f.write(f"  Median:     {np.median(approx_entropies_clean):>10.4f}\n")
            f.write(f"  Std Dev:    {np.std(approx_entropies_clean):>10.4f}\n")
            f.write(f"  Min:        {np.min(approx_entropies_clean):>10.4f}\n")
            f.write(f"  Max:        {np.max(approx_entropies_clean):>10.4f}\n\n")
        
        # Information coupling statistics
        f.write("\nINFORMATION COUPLING WITH ENSO\n")
        f.write("-"*100 + "\n\n")
        
        mi_values = [r['mi_enso_pitch'] for r in modes_with_data]
        nmi_values = [r['nmi_enso_pitch'] for r in modes_with_data]
        
        enso_entropy, _, _ = shannon_entropy(enso_data)
        preservation_values = [(mi / enso_entropy * 100) for mi in mi_values]
        
        f.write("Mutual Information (ENSO, Pitch) [bits]\n")
        f.write(f"  Mean:       {np.mean(mi_values):>10.4f}\n")
        f.write(f"  Median:     {np.median(mi_values):>10.4f}\n")
        f.write(f"  Std Dev:    {np.std(mi_values):>10.4f}\n")
        f.write(f"  Min:        {np.min(mi_values):>10.4f}\n")
        f.write(f"  Max:        {np.max(mi_values):>10.4f}\n\n")
        
        f.write("Normalized Mutual Information [0-1]\n")
        f.write(f"  Mean:       {np.mean(nmi_values):>10.4f}\n")
        f.write(f"  Median:     {np.median(nmi_values):>10.4f}\n")
        f.write(f"  Std Dev:    {np.std(nmi_values):>10.4f}\n")
        f.write(f"  Min:        {np.min(nmi_values):>10.4f}\n")
        f.write(f"  Max:        {np.max(nmi_values):>10.4f}\n\n")
        
        f.write("Information Preservation [%]\n")
        f.write(f"  Mean:       {np.mean(preservation_values):>10.2f}%\n")
        f.write(f"  Median:     {np.median(preservation_values):>10.2f}%\n")
        f.write(f"  Std Dev:    {np.std(preservation_values):>10.2f}%\n")
        f.write(f"  Min:        {np.min(preservation_values):>10.2f}%\n")
        f.write(f"  Max:        {np.max(preservation_values):>10.2f}%\n\n")
        
        # Comparison by scale
        f.write("\nCOMPARISON BY GAMELAN SCALE\n")
        f.write("-"*100 + "\n\n")
        
        pelog_modes = [r for r in modes_with_data if 'pelog' in r['mode']]
        slendro_modes = [r for r in modes_with_data if 'slendro' in r['mode']]
        
        if len(pelog_modes) > 0:
            pelog_mi = [r['mi_enso_pitch'] for r in pelog_modes]
            pelog_pres = [(r['mi_enso_pitch'] / enso_entropy * 100) for r in pelog_modes]
            
            f.write("Pelog Scale:\n")
            f.write(f"  Number of modes:                   {len(pelog_modes)}\n")
            f.write(f"  Average MI (ENSO, Pitch):          {np.mean(pelog_mi):.4f} bits\n")
            f.write(f"  Average Information Preservation:  {np.mean(pelog_pres):.2f}%\n\n")
        
        if len(slendro_modes) > 0:
            slendro_mi = [r['mi_enso_pitch'] for r in slendro_modes]
            slendro_pres = [(r['mi_enso_pitch'] / enso_entropy * 100) for r in slendro_modes]
            
            f.write("Slendro Scale:\n")
            f.write(f"  Number of modes:                   {len(slendro_modes)}\n")
            f.write(f"  Average MI (ENSO, Pitch):          {np.mean(slendro_mi):.4f} bits\n")
            f.write(f"  Average Information Preservation:  {np.mean(slendro_pres):.2f}%\n\n")
        
        # Comparison by composition type
        f.write("\nCOMPARISON BY COMPOSITION TYPE\n")
        f.write("-"*100 + "\n\n")
        
        for comp_type in ['layered', 'alternating', 'melodic', 'spectral']:
            comp_modes = [r for r in modes_with_data if comp_type in r['mode']]
            if len(comp_modes) > 0:
                comp_mi = [r['mi_enso_pitch'] for r in comp_modes]
                comp_pres = [(r['mi_enso_pitch'] / enso_entropy * 100) for r in comp_modes]
                
                f.write(f"{comp_type.capitalize()}:\n")
                f.write(f"  Number of modes:                   {len(comp_modes)}\n")
                f.write(f"  Average MI (ENSO, Pitch):          {np.mean(comp_mi):.4f} bits\n")
                f.write(f"  Average Information Preservation:  {np.mean(comp_pres):.2f}%\n\n")
        
        # Ranking
        f.write("\nRANKING BY INFORMATION PRESERVATION\n")
        f.write("-"*100 + "\n\n")
        
        sorted_modes = sorted(modes_with_data, 
                            key=lambda x: x['mi_enso_pitch'], 
                            reverse=True)
        
        f.write(f"{'Rank':<6} {'Mode':<30} {'MI [bits]':<12} {'Preservation [%]':<15}\n")
        f.write("-"*100 + "\n")
        
        for i, r in enumerate(sorted_modes, 1):
            mi = r['mi_enso_pitch']
            pres = (mi / enso_entropy * 100)
            f.write(f"{i:<6} {r['mode']:<30} {mi:>8.4f}    {pres:>10.2f}\n")
        
        f.write("\n")
        f.write("="*100 + "\n")
        f.write("END OF DESCRIPTIVE STATISTICS\n")
        f.write("="*100 + "\n")
    
    print(f"    Saved: {output_file.name}")


def main():
    print("\n" + "="*70)
    print("MUSIC ANALYSIS RESULTS EXTRACTOR")
    print("="*70 + "\n")
    
    STATS_DIR.mkdir(exist_ok=True)
    
    # Load data
    print("Loading data...")
    if not PICKLE_FILE.exists():
        print(f"  Error: {PICKLE_FILE} not found!")
        print("  Run music_analysis_parallel.py first to generate results.")
        return
    
    with open(PICKLE_FILE, 'rb') as f:
        all_results = pickle.load(f)
    
    enso_data = load_enso_data(INPUT_FILE)
    
    print(f"  Loaded {len(all_results)} mode results")
    print(f"  Loaded {len(enso_data)} months of ENSO data\n")
    
    # Generate outputs
    print("Generating outputs...")
    create_summary_csv(all_results, enso_data, OUTPUT_SUMMARY_CSV)
    create_detailed_csv(all_results, OUTPUT_DETAILED_CSV)
    write_entropy_calculations(all_results, enso_data, OUTPUT_ENTROPY_TXT)
    write_descriptive_statistics(all_results, enso_data, OUTPUT_STATISTICS_TXT)
    
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  - {OUTPUT_SUMMARY_CSV.name}")
    print(f"  - {OUTPUT_DETAILED_CSV.name}")
    print(f"  - {OUTPUT_ENTROPY_TXT.name}")
    print(f"  - {OUTPUT_STATISTICS_TXT.name}")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
