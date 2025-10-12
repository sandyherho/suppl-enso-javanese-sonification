#!/usr/bin/env python
"""
enso_descriptive_stats.py

Comprehensive Descriptive Statistics for ENSO Nino 3.4 Index
============================================================

Authors: Sandy H. S. Herho, Rusmawan Suwarman, Edi Riawan, Nurjanna J. Trilaksono
Institution: Weather and Climate Prediction Laboratory (WCPL) ITB
Date: 10/11/2025
License: WTFPL
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

# Configuration
INPUT_DIR = Path('../input_data')
STATS_DIR = Path('../stats')
INPUT_FILE = INPUT_DIR / 'nino34_hadisst_mon_1870_2024.csv'
OUTPUT_FILE = STATS_DIR / 'enso_descriptive_statistics.txt'


def load_enso_data(filepath):
    """Load ENSO Nino 3.4 data."""
    df = pd.read_csv(filepath, skipinitialspace=True)
    df.columns = df.columns.str.strip().str.lower()
    df['date'] = pd.to_datetime(df['date'])
    df['nino34'] = pd.to_numeric(df['nino34'], errors='coerce')
    df.loc[df['nino34'] < -900, 'nino34'] = np.nan
    df = df.dropna(subset=['nino34'])
    return df['date'].values, df['nino34'].values


def find_extreme_events(dates, data, threshold=1.5):
    """Find all extreme El Niño and La Niña events."""
    dates_dt = pd.to_datetime(dates)
    
    el_nino_events = []
    la_nina_events = []
    
    # Find strong El Niño events (>= threshold)
    strong_el_nino = data >= threshold
    if np.any(strong_el_nino):
        indices = np.where(strong_el_nino)[0]
        for idx in indices:
            el_nino_events.append({
                'date': dates_dt[idx],
                'value': data[idx],
                'type': 'Strong El Niño' if data[idx] >= 2.0 else 'Moderate El Niño'
            })
    
    # Find strong La Niña events (<= -threshold)
    strong_la_nina = data <= -threshold
    if np.any(strong_la_nina):
        indices = np.where(strong_la_nina)[0]
        for idx in indices:
            la_nina_events.append({
                'date': dates_dt[idx],
                'value': data[idx],
                'type': 'Strong La Niña' if data[idx] <= -2.0 else 'Moderate La Niña'
            })
    
    return el_nino_events, la_nina_events


def identify_enso_episodes(dates, data):
    """Identify continuous ENSO episodes (3+ consecutive months above/below threshold)."""
    dates_dt = pd.to_datetime(dates)
    episodes = []
    
    # El Niño episodes (>= 0.5 for 3+ months)
    current_episode = None
    for i, val in enumerate(data):
        if val >= 0.5:
            if current_episode is None:
                current_episode = {
                    'type': 'El Niño',
                    'start_idx': i,
                    'values': [val]
                }
            else:
                current_episode['values'].append(val)
        else:
            if current_episode is not None and len(current_episode['values']) >= 3:
                current_episode['end_idx'] = i - 1
                current_episode['start_date'] = dates_dt[current_episode['start_idx']]
                current_episode['end_date'] = dates_dt[current_episode['end_idx']]
                current_episode['peak_value'] = max(current_episode['values'])
                current_episode['mean_value'] = np.mean(current_episode['values'])
                current_episode['duration'] = len(current_episode['values'])
                episodes.append(current_episode)
            current_episode = None
    
    # Check last episode
    if current_episode is not None and len(current_episode['values']) >= 3:
        current_episode['end_idx'] = len(data) - 1
        current_episode['start_date'] = dates_dt[current_episode['start_idx']]
        current_episode['end_date'] = dates_dt[current_episode['end_idx']]
        current_episode['peak_value'] = max(current_episode['values'])
        current_episode['mean_value'] = np.mean(current_episode['values'])
        current_episode['duration'] = len(current_episode['values'])
        episodes.append(current_episode)
    
    # La Niña episodes (<= -0.5 for 3+ months)
    current_episode = None
    for i, val in enumerate(data):
        if val <= -0.5:
            if current_episode is None:
                current_episode = {
                    'type': 'La Niña',
                    'start_idx': i,
                    'values': [val]
                }
            else:
                current_episode['values'].append(val)
        else:
            if current_episode is not None and len(current_episode['values']) >= 3:
                current_episode['end_idx'] = i - 1
                current_episode['start_date'] = dates_dt[current_episode['start_idx']]
                current_episode['end_date'] = dates_dt[current_episode['end_idx']]
                current_episode['peak_value'] = min(current_episode['values'])
                current_episode['mean_value'] = np.mean(current_episode['values'])
                current_episode['duration'] = len(current_episode['values'])
                episodes.append(current_episode)
            current_episode = None
    
    # Check last episode
    if current_episode is not None and len(current_episode['values']) >= 3:
        current_episode['end_idx'] = len(data) - 1
        current_episode['start_date'] = dates_dt[current_episode['start_idx']]
        current_episode['end_date'] = dates_dt[current_episode['end_idx']]
        current_episode['peak_value'] = min(current_episode['values'])
        current_episode['mean_value'] = np.mean(current_episode['values'])
        current_episode['duration'] = len(current_episode['values'])
        episodes.append(current_episode)
    
    # Sort by start date
    episodes.sort(key=lambda x: x['start_date'])
    
    return episodes


def write_descriptive_stats(dates, enso_data, output_file):
    """Write comprehensive ENSO descriptive statistics to file."""
    dates_dt = pd.to_datetime(dates)
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ENSO NINO 3.4 INDEX - COMPREHENSIVE DESCRIPTIVE STATISTICS\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Authors: Sandy H. S. Herho, Rusmawan Suwarman, Edi Riawan, Nurjanna J. Trilaksono\n")
        f.write("Institution: Weather and Climate Prediction Laboratory (WCPL) ITB\n")
        f.write("="*80 + "\n\n")
        
        # Dataset period
        f.write("DATASET PERIOD\n")
        f.write("-"*80 + "\n")
        f.write(f"Start Date:    {dates_dt[0].strftime('%B %Y')}\n")
        f.write(f"End Date:      {dates_dt[-1].strftime('%B %Y')}\n")
        f.write(f"Total Months:  {len(enso_data)}\n")
        f.write(f"Total Years:   {len(enso_data)/12:.1f}\n\n")
        
        # Basic statistics
        f.write("CENTRAL TENDENCY AND DISPERSION\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean:          {np.mean(enso_data):>8.4f} C\n")
        f.write(f"Median:        {np.median(enso_data):>8.4f} C\n")
        f.write(f"Std Deviation: {np.std(enso_data):>8.4f} C\n")
        f.write(f"Variance:      {np.var(enso_data):>8.4f} C^2\n")
        f.write(f"Range:         {np.max(enso_data) - np.min(enso_data):>8.4f} C\n\n")
        
        # Extremes
        max_idx = np.argmax(enso_data)
        min_idx = np.argmin(enso_data)
        f.write("EXTREME VALUES\n")
        f.write("-"*80 + "\n")
        f.write(f"Maximum:       {np.max(enso_data):>8.4f} C on {dates_dt[max_idx].strftime('%B %Y')}\n")
        f.write(f"Minimum:       {np.min(enso_data):>8.4f} C on {dates_dt[min_idx].strftime('%B %Y')}\n\n")
        
        # Percentiles
        f.write("PERCENTILE DISTRIBUTION\n")
        f.write("-"*80 + "\n")
        for p in [5, 10, 25, 50, 75, 90, 95]:
            f.write(f"{p:>2d}th percentile: {np.percentile(enso_data, p):>8.4f} C\n")
        f.write(f"IQR (Q3-Q1):     {np.percentile(enso_data, 75) - np.percentile(enso_data, 25):>8.4f} C\n\n")
        
        # Distribution shape
        skew = stats.skew(enso_data)
        kurt = stats.kurtosis(enso_data)
        f.write("DISTRIBUTION SHAPE\n")
        f.write("-"*80 + "\n")
        f.write(f"Skewness:      {skew:>8.4f}\n")
        f.write(f"Kurtosis:      {kurt:>8.4f}\n\n")
        
        # ENSO phases
        el_nino_mask = enso_data >= 0.5
        la_nina_mask = enso_data <= -0.5
        neutral_mask = (enso_data > -0.5) & (enso_data < 0.5)
        
        f.write("ENSO PHASE STATISTICS\n")
        f.write("-"*80 + "\n\n")
        
        f.write(f"El Nino (>= 0.5 C):\n")
        f.write(f"  Months:        {np.sum(el_nino_mask):>6d} ({100*np.sum(el_nino_mask)/len(enso_data):>5.2f}%)\n")
        if np.any(el_nino_mask):
            f.write(f"  Mean:          {np.mean(enso_data[el_nino_mask]):>8.4f} C\n")
            f.write(f"  Maximum:       {np.max(enso_data[el_nino_mask]):>8.4f} C\n")
        f.write("\n")
        
        f.write(f"La Nina (<= -0.5 C):\n")
        f.write(f"  Months:        {np.sum(la_nina_mask):>6d} ({100*np.sum(la_nina_mask)/len(enso_data):>5.2f}%)\n")
        if np.any(la_nina_mask):
            f.write(f"  Mean:          {np.mean(enso_data[la_nina_mask]):>8.4f} C\n")
            f.write(f"  Minimum:       {np.min(enso_data[la_nina_mask]):>8.4f} C\n")
        f.write("\n")
        
        f.write(f"Neutral (-0.5 to 0.5 C):\n")
        f.write(f"  Months:        {np.sum(neutral_mask):>6d} ({100*np.sum(neutral_mask)/len(enso_data):>5.2f}%)\n\n")
        
        # Extreme events listing
        f.write("\n" + "="*80 + "\n")
        f.write("EXTREME EVENTS LISTING\n")
        f.write("="*80 + "\n\n")
        
        el_nino_events, la_nina_events = find_extreme_events(dates, enso_data, threshold=1.5)
        
        f.write(f"STRONG EL NIÑO EVENTS (>= 1.5 C): {len(el_nino_events)} occurrences\n")
        f.write("-"*80 + "\n")
        if el_nino_events:
            f.write(f"{'Date':<15} {'Value (C)':<12} {'Category':<20}\n")
            f.write("-"*80 + "\n")
            for event in el_nino_events:
                f.write(f"{event['date'].strftime('%B %Y'):<15} {event['value']:>8.3f}     {event['type']:<20}\n")
        else:
            f.write("No strong El Niño events recorded.\n")
        f.write("\n")
        
        f.write(f"STRONG LA NIÑA EVENTS (<= -1.5 C): {len(la_nina_events)} occurrences\n")
        f.write("-"*80 + "\n")
        if la_nina_events:
            f.write(f"{'Date':<15} {'Value (C)':<12} {'Category':<20}\n")
            f.write("-"*80 + "\n")
            for event in la_nina_events:
                f.write(f"{event['date'].strftime('%B %Y'):<15} {event['value']:>8.3f}     {event['type']:<20}\n")
        else:
            f.write("No strong La Niña events recorded.\n")
        f.write("\n")
        
        # ENSO Episodes
        f.write("\n" + "="*80 + "\n")
        f.write("ENSO EPISODES (3+ consecutive months above/below threshold)\n")
        f.write("="*80 + "\n\n")
        
        episodes = identify_enso_episodes(dates, enso_data)
        
        el_nino_episodes = [e for e in episodes if e['type'] == 'El Niño']
        la_nina_episodes = [e for e in episodes if e['type'] == 'La Niña']
        
        f.write(f"EL NIÑO EPISODES: {len(el_nino_episodes)}\n")
        f.write("-"*80 + "\n")
        if el_nino_episodes:
            f.write(f"{'Start':<12} {'End':<12} {'Duration':<10} {'Peak':<10} {'Mean':<10}\n")
            f.write(f"{'Date':<12} {'Date':<12} {'(months)':<10} {'(C)':<10} {'(C)':<10}\n")
            f.write("-"*80 + "\n")
            for ep in el_nino_episodes:
                f.write(f"{ep['start_date'].strftime('%b %Y'):<12} "
                       f"{ep['end_date'].strftime('%b %Y'):<12} "
                       f"{ep['duration']:<10} "
                       f"{ep['peak_value']:>8.3f}  "
                       f"{ep['mean_value']:>8.3f}\n")
        f.write("\n")
        
        f.write(f"LA NIÑA EPISODES: {len(la_nina_episodes)}\n")
        f.write("-"*80 + "\n")
        if la_nina_episodes:
            f.write(f"{'Start':<12} {'End':<12} {'Duration':<10} {'Peak':<10} {'Mean':<10}\n")
            f.write(f"{'Date':<12} {'Date':<12} {'(months)':<10} {'(C)':<10} {'(C)':<10}\n")
            f.write("-"*80 + "\n")
            for ep in la_nina_episodes:
                f.write(f"{ep['start_date'].strftime('%b %Y'):<12} "
                       f"{ep['end_date'].strftime('%b %Y'):<12} "
                       f"{ep['duration']:<10} "
                       f"{ep['peak_value']:>8.3f}  "
                       f"{ep['mean_value']:>8.3f}\n")
        f.write("\n")
        
        # Temporal characteristics
        rate_of_change = np.diff(enso_data)
        roc_max_idx = np.argmax(rate_of_change)
        roc_min_idx = np.argmin(rate_of_change)
        
        f.write("RATE OF CHANGE STATISTICS (Month-to-Month)\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean ROC:      {np.mean(rate_of_change):>8.4f} C/month\n")
        f.write(f"Std Dev ROC:   {np.std(rate_of_change):>8.4f} C/month\n")
        f.write(f"Max Increase:  {np.max(rate_of_change):>8.4f} C/month on {dates_dt[roc_max_idx].strftime('%B %Y')}\n")
        f.write(f"Max Decrease:  {np.min(rate_of_change):>8.4f} C/month on {dates_dt[roc_min_idx].strftime('%B %Y')}\n\n")
        
        # Autocorrelation
        autocorr = np.corrcoef(enso_data[:-1], enso_data[1:])[0, 1]
        f.write("TEMPORAL CHARACTERISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Lag-1 Autocorrelation: {autocorr:>6.4f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF DESCRIPTIVE STATISTICS\n")
        f.write("="*80 + "\n")
    
    print(f"  Saved: {output_file.name}")


def main():
    print("\n" + "="*70)
    print("ENSO DESCRIPTIVE STATISTICS")
    print("="*70 + "\n")
    
    STATS_DIR.mkdir(exist_ok=True)
    
    print("Loading ENSO data...")
    dates, enso_data = load_enso_data(INPUT_FILE)
    print(f"  Loaded {len(enso_data)} months of data\n")
    
    print("Calculating statistics...")
    write_descriptive_stats(dates, enso_data, OUTPUT_FILE)
    
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
