================================================================================
MUSICAL STATE VARIABLES
================================================================================

This file contains the two primary state variables for each musical mode:

1. TONAL CENTER (MIDI note number)
   - Weighted mean pitch across the entire piece
   - Represents the musical 'home base'
   - Higher values = higher overall pitch

2. TENSION INDEX (0-1 scale)
   - Combined measure of:
     * Velocity (40% weight)
     * Note density (30% weight)
     * Pitch spread (30% weight)
   - 0 = minimal tension (calm)
   - 1 = maximal tension (intense)

USAGE FOR VISUALIZATION
--------------------------------------------------------------------------------

2D State Space Plot:
  df = pd.read_csv('music_state_variables.csv')
  plt.scatter(df['tonal_center'], df['tension_index'])
  plt.xlabel('Tonal Center (MIDI note)')
  plt.ylabel('Tension Index')

Compare by Scale:
  pelog = df[df['scale_family'] == 'Pelog']
  slendro = df[df['scale_family'] == 'Slendro']

Compare by Composition Type:
  for ctype in df['composition_type'].unique():
      subset = df[df['composition_type'] == ctype]
      plt.scatter(subset['tonal_center'], subset['tension_index'], label=ctype)

================================================================================
