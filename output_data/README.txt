================================================================================
MUSIC ANALYSIS OUTPUT DATA
================================================================================

TIME SERIES FILES
--------------------------------------------------------------------------------

Each music_timeseries_*.csv contains:

Columns:
  time_step                    : Sequential integer (0, 1, 2, ...)
  time_beats                   : Time in musical beats
  pitch_center_mass            : Weighted average pitch (raw)
  velocity_mean                : Average velocity of active notes
  note_density                 : Number of simultaneous notes
  pitch_variance               : Spread of pitches
  pitch_center_mass_smoothed   : Smoothed PCM (for plotting)
  pitch_center_mass_derivative : Rate of change of PCM

PRIMARY VARIABLE: pitch_center_mass_smoothed

MATHEMATICAL DEFINITION
--------------------------------------------------------------------------------

Pitch Center of Mass (PCM):

  PCM(t) = Σ[pitch_i(t) × velocity_i(t)] / Σ[velocity_i(t)]

This represents the 'musical state' as a single continuous value,
analogous to center of mass in physics.

USAGE FOR PHASE SPACE ANALYSIS
--------------------------------------------------------------------------------

Plot ENSO vs Musical State:
  plt.scatter(enso, music['pitch_center_mass_smoothed'])
  plt.xlabel('ENSO Index (°C)')
  plt.ylabel('Musical State (Pitch Center of Mass)')

================================================================================
Generated: 2025-10-12 03:00:07
================================================================================
