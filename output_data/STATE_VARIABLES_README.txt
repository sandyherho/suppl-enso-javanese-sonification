================================================================================
REVISED: DYNAMIC MUSICAL STATE VARIABLES WITH GLOBAL NORMALIZATION
================================================================================

IMPROVEMENTS IN THIS VERSION:
  1. Global normalization - all modes on same scale
  2. Increased smoothing - window=20 (was 10)
  3. Better hop_length - 1024 samples (was 512)
  4. Percentile-based normalization (robust to outliers)

NORMALIZATION PARAMETERS:
  Spectral Centroid: 436.4 - 3095.1 Hz
  RMS Energy: 0 - 0.033531

USE THESE COLUMNS FOR PHASE SPACE PLOTS:
  - spectral_centroid_smooth_norm (0-1 scale, globally normalized)
  - rms_energy_smooth_norm (0-1 scale, globally normalized)

================================================================================
