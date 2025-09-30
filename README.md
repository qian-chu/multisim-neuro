# MultiSim

**Simulate realistic MEG/EEG data with ground-truth multivariate effects**

## Installation

```bash
pip install multisim-neuro
```

Or clone and install the development version:

```bash
git clone https://github.com/AlexLepauvre/multisim-neuro
cd multisim-neuro
pip install -e .[full]
```

## Features

- Design-driven simulation: specify arbitrary experimental design matrices (conditions, interactions, parametric regressors).

- Time‑locked effects: inject multivariate patterns in chosen time windows with exact control over effect size (Mahalanobis distance).

- Noise modeling: control within‑subject sensor noise and between‑subject variability.

- Spatial covariance: simulate correlated sensors or modes via any user‑supplied covariance matrix.

- MNE & EEGLAB export: seamless conversion to mne.EpochsArray or EEGLAB .set files.

- Power analysis: predict group‑level statistical power from trials, subjects, and effect sizes.

## Quickstart
```python
import numpy as np
from multisim import Simulator

# 1. Create a simple design: two conditions, 100 trials
X = np.vstack([np.zeros(100), np.ones(100)]).T  
t_win   = np.array([[0.1, 0.3]])    # effect between 100–300 ms
effects = np.array([1])             # effect on condition 1

# 2. Instantiate simulator
sim = Simulator(
    X, noise_std=1.0, n_channels=64, n_subjects=10,
    tmin=-0.2, tmax=0.8, sfreq=250,
    t_win=t_win, effects=effects,
    effect_size=[0.5],               # multivariate d′ = 0.5
    intersub_noise_std=0.1           # between-subject σ
)

# 3. Export to MNE Epochs and decode
epochs_list = sim.export_to_mne()
# ... run your decoding pipeline on epochs_list ...
```

## API
After installation, see full class and method documentation at: https://alexlepauvre.github.io/multisim-neuro/index.html

You can find extensive tutorials at:
https://alexlepauvre.github.io/multisim-neuro/tutorial/index.html

## Customization

- Spatial pattern: pass a custom covariance matrix or weight vector to concentrate signal in subsets of channels.

- Temporal kernel: supply any causal kernel to shape the time-course of effects.

- Effect size: Simulate multivariate effect at particular effect size to test the ability of your pipelines to retrieve it given the number of subjects and trials for each subject

## How to cite us:
If you use the scripts found in this repository, you can use the DOI provided by Zenodo to cite us. And here is a bibtex:

```
@article{LepauvreEtAl2024,
  title = {MultiSim},
  author = {Lepauvre, Alex and Chu, Qian and Zeidman, Peter and Melloni, Lucia},
  year = {2025},
  doi = {https://doi.org/10.5281/zenodo.17231750},
}
```