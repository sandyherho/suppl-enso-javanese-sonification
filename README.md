# Supplementary Materials: Preliminary sonification of ENSO using traditional Javanese gamelan scales

[![DOI](https://zenodo.org/badge/1074607592.svg)](https://doi.org/10.5281/zenodo.17333649)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-013243.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-150458.svg)](https://pandas.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7%2B-8CAAE6.svg)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-11557c.svg)](https://matplotlib.org/)
[![Librosa](https://img.shields.io/badge/Librosa-0.9%2B-orange.svg)](https://librosa.org/)
[![MIDIUtil](https://img.shields.io/badge/MIDIUtil-1.2%2B-green.svg)](https://github.com/MarkCWirt/MIDIUtil)
[![License](https://img.shields.io/badge/License-WTFPL-red.svg)](http://www.wtfpl.net/)


## Overview

This repository contains supplementary materials for the manuscript on sonifying El Niño-Southern Oscillation (ENSO) dynamics using traditional Javanese gamelan scales. The project transforms 155 years of Niño 3.4 index data (1870-2024) into musical compositions using Pelog and Slendro scales, with subsequent analysis of the musical characteristics and phase space dynamics.


## Data

**ENSO Data Source:** HadISST 1.1 monthly Niño 3.4 index (1870-2024)  
Source: https://psl.noaa.gov/data/timeseries/month/DS/Nino34/

**Musical Sonification Outputs (MIDI & WAV):** Available at OSF: https://doi.org/10.17605/OSF.IO/QY82M

## Methodology

### Sonification Strategy

**Scales:** Traditional Javanese Pelog and Slendro with interval sets:

$$\Phi_{\text{Pelog}} = \{0, 1, 3, 7, 8\} \quad \text{semitones}$$

$$\Phi_{\text{Slendro}} = \{0, 2, 3, 7, 9\} \quad \text{semitones}$$

**Pitch Mapping:** Let $\theta(t) \in [-3, 3]$ °C denote the ENSO anomaly at time $t$. Normalization:

$$\xi(t) = \frac{\theta(t) + 3}{6}, \quad \xi \in [0, 1]$$

Continuous position across $n_{\text{oct}} = 2$ octaves spanning $|\Phi| = 5$ scale degrees:

$$\psi(t) = \xi(t) \cdot (|\Phi| \cdot n_{\text{oct}} - 1)$$

Quantized step and MIDI note assignment:

$$s(t) = \lfloor \psi(t) + 0.5 \rfloor, \quad s \in [0, 9]$$

$$\nu(t) = 60 + 12 \left\lfloor \frac{s(t)}{|\Phi|} \right\rfloor + \Phi_{s(t) \bmod |\Phi|}$$

where $\nu(t) \in [0, 127]$ is the MIDI note number.

**Velocity Encoding:** Combined magnitude and rate-of-change dynamics:

$$v(t) = \text{clip}\left(\beta_0 + \beta_1|\theta(t)| + \beta_2\left|\frac{d\theta}{dt}\right|, v_{\min}, v_{\max}\right)$$

with $\beta_0 = 80$, $\beta_1 = 20$, $\beta_2 = 30$, $v_{\min} = 40$, $v_{\max} = 127$.

**Multi-scale Temporal Decomposition:** Rolling mean with window $\omega \in \{3, 12, 24, 36\}$ months:

$$\bar{\theta}_\omega(t) = \frac{1}{\omega}\sum_{k=0}^{\omega-1} \theta(t-k)$$

### Musical Feature Extraction

**Spectral Centroid (Brightness):** For discrete-time signal $x[n]$ with STFT $X[k, m]$:

$$\mathcal{C}[m] = \frac{\displaystyle\sum_{k=0}^{K-1} f[k] \cdot |X[k, m]|}{\displaystyle\sum_{k=0}^{K-1} |X[k, m]|}$$

where $f[k] = k \cdot f_s / N$ is the frequency bin center, $f_s$ is sampling rate, $N$ is FFT size, and $m$ is the frame index.

**RMS Energy (Intensity):** For frame $m$ with hop length $H$:

$$\mathcal{E}[m] = \sqrt{\frac{1}{L}\sum_{n=0}^{L-1} x^2[mH + n]}$$

where $L$ is the frame length.

**Global Normalization:** To ensure comparability across modes, normalize using global statistics:

$$\mathcal{C}_{\text{norm}}[m] = \frac{\mathcal{C}[m] - \mu_{\mathcal{C}}^{\text{global}}}{\sigma_{\mathcal{C}}^{\text{global}}}, \quad \mathcal{E}_{\text{norm}}[m] = \frac{\mathcal{E}[m]}{\max(\mathcal{E})^{\text{global}}}$$

**Temporal Smoothing:** Apply moving average with window $\omega_s = 20$ frames:

$$\tilde{\mathcal{C}}[m] = \frac{1}{\omega_s}\sum_{j=-\lfloor\omega_s/2\rfloor}^{\lfloor\omega_s/2\rfloor} \mathcal{C}[m + j]$$

**Dynamic Range:**

$$\Delta_{\mathcal{E}} = \frac{\max(\mathcal{E})}{\bar{\mathcal{E}}}$$

**Coefficient of Variation:**

$$\gamma = \frac{\sigma}{\mu}$$

where $\sigma$ and $\mu$ are standard deviation and mean, respectively.

**Lag-1 Autocorrelation:** For time series $y[m]$:

$$\rho_1 = \frac{\displaystyle\sum_{m=1}^{M-1}(y[m] - \bar{y})(y[m+1] - \bar{y})}{\displaystyle\sum_{m=1}^{M}(y[m] - \bar{y})^2}$$

### Phase Space Analysis

Define phase space trajectory as:

$$\Gamma(t) = (\mathcal{C}_{\text{norm}}(t), \mathcal{E}_{\text{norm}}(t)) \in [0,1]^2$$

**Convex Hull Area:**

$$\mathcal{A} = \text{Area}\left(\text{ConvexHull}\{\Gamma(t_i)\}_{i=1}^M\right)$$

**Trajectory Path Length:**

$$\mathcal{L} = \sum_{i=1}^{M-1} \|\Gamma(t_{i+1}) - \Gamma(t_i)\|_2$$

where $\|\cdot\|_2$ denotes Euclidean distance.

**Exploration Efficiency Index:**

$$\eta = \frac{\mathcal{A}}{\mathcal{L}}$$

Higher $\eta$ indicates efficient space exploration; lower $\eta$ suggests meandering trajectories.

**Trajectory Centroid:**

$$\bar{\Gamma} = \left(\bar{\mathcal{C}}, \bar{\mathcal{E}}\right) = \frac{1}{M}\sum_{i=1}^{M}\Gamma(t_i)$$

**Trajectory Spread:**

$$\sigma_{\Gamma} = \left(\sigma_{\mathcal{C}}, \sigma_{\mathcal{E}}\right)$$

**Pearson Cross-Correlation:**

$$\rho_{\mathcal{C}\mathcal{E}} = \frac{\text{Cov}(\mathcal{C}, \mathcal{E})}{\sigma_{\mathcal{C}} \sigma_{\mathcal{E}}} = \frac{\displaystyle\sum_{i=1}^{M}(\mathcal{C}_i - \bar{\mathcal{C}})(\mathcal{E}_i - \bar{\mathcal{E}})}{\sqrt{\displaystyle\sum_{i=1}^{M}(\mathcal{C}_i - \bar{\mathcal{C}})^2}\sqrt{\displaystyle\sum_{i=1}^{M}(\mathcal{E}_i - \bar{\mathcal{E}})^2}}$$

**Revisit Rate:** Fraction of trajectory pairs within threshold $\epsilon = 0.05$:

$$\mathcal{R}_\epsilon = \frac{1}{N_{\text{pairs}}}\sum_{i<j, |j-i|>5} \mathbb{1}(\|\Gamma(t_i) - \Gamma(t_j)\|_2 < \epsilon)$$

where $\mathbb{1}(\cdot)$ is the indicator function.

## Requirements

### Python Libraries

```bash
pip install numpy pandas scipy matplotlib librosa midiutil soundfile
```

### External Software (Optional)

**For MIDI-to-WAV Conversion:**  
FluidSynth is required to convert MIDI files to WAV audio. This is not a Python library.

- **Linux/macOS:**
  ```bash
  # Ubuntu/Debian
  sudo apt-get install fluidsynth fluid-soundfont-gm
  
  # macOS
  brew install fluidsynth
  ```

- **Windows:**  
  Download from [FluidSynth releases](https://github.com/FluidSynth/fluidsynth/releases)

The script will generate MIDI files without FluidSynth, but WAV conversion will be skipped if FluidSynth is not available.

## Usage

Execute scripts in order:

```bash
cd script
python enso_descriptive_stats.py      # Generate ENSO statistics
python enso_sonification.py           # Create MIDI/WAV files
python enso_ts_plot.py                # Plot time series
python music_characterization.py      # Extract musical features
python music_state_ts_analysis.py     # Analyze state variables
python phase_space_analysis.py        # Phase space analysis
```

## Key Results

- **8 sonification variants** (2 scales × 4 composition modes)
- **Phase space coverage:** $\mathcal{A} \in [0.211, 0.529]$ (normalized units)
- **Brightness range:** $\mathcal{C} \in [809, 1863]$ Hz
- **Temporal persistence:** $\rho_1 \in [0.95, 0.99]$
- **Brightness-energy coupling:** Predominantly $\rho_{\mathcal{C}\mathcal{E}} < 0$ (inverse relationship)

Detailed results in `stats/` directory.

## Citation

If you use this code or data, please cite:

```bibtex
@article{herho2025enso,
  title={Preliminary sonification of {ENSO} using traditional {Javanese} gamelan scales},
  author={Herho, Sandy H. S. and Suwarman, Rusmawan and Trilaksono, Nurjanna J., Anwar, Iwan P.},
  journal={xxx},
  year={202x},
  volume={xxx},
  pages={xxx},
  doi={xxxx}
}
```

## Authors

**Sandy H. S. Herho**, **Rusmawan Suwarman**, **Nurjanna J. Trilaksono, Iwan P. Anwar**  


## License

WTFPL - Do What The F*** You Want To Public License

