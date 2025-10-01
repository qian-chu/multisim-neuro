# MultiSim: A Python Toolbox for Simulating Datasets with Time-Resolved Multivariate Effects

## Motivation and Overview

One critical challenge in designing analysis pipelines for MEG/EEG data is confirming that the pipeline is both sensitive (i.e., it detects real effects) and specific (i.e., it avoids false alarms). In most cases, it is not known a priori whether or when experimental effects are present in the data, and it is therefore not possible to assess the sensitivity and specificity of analysis pipelines based on the data one is trying to analyse.

**MultiSim** is a Python package for simulating multivariate EEG/MEG datasets with user-defined experimental effects.  
It enables principled testing and validation of decoding pipelines, source reconstruction methods, and statistical analyses.

Specifically, the toolbox allows to:

- Specify a between-trial design (e.g., two conditions, Condition A and Condition B).
- Inject multivariate effects at particular time windows (e.g., Condition A is active from 100–200 ms, Condition B from 300–400 ms).
- Control signal-to-noise ratio, spatial covariance, temporal smoothing, and between-subject variability
- Generate multiple subjects for group-level statistical analysis
- Export to [MNE](https://mne.tools/) and [EEGLAB](https://sccn.ucsd.edu/eeglab/) formats
- Validate that the pipeline recovers the known effects accurately.

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

## Usage Example

```python
import numpy as np
import pandas as pd
from multisim import Simulator

# Define experimental design: 100 trials, 1 condition
X = pd.DataFrame(np.random.randn(100, 1), columns=["category"])
effects = [{"condition": "category", "windows": [0.1, 0.3], "effect_size": 0.5}]

# Simulate data
sim = Simulator(
    X,
    effects,
    noise_std=0.1,
    n_channels=64,
    n_subjects=20,
    tmin=-0.2,
    tmax=0.8,
    sfreq=250,
)
print(sim) # Overview of the simulation parameters
first_subject_data = sim.data[0]  # Access data for the first subject
```

## Documentation

Full class and method documentation are available at: <https://alexlepauvre.github.io/multisim-neuro/index.html>.

Tutorials are available at: <https://alexlepauvre.github.io/multisim-neuro/tutorial/index.html>

## License

MultiSim is licensed under the MIT License.

## Citation

If you use the scripts found in this repository, you can use the DOI provided by Zenodo to cite us. And here is a bibtex:

```bibtex
@article{LepauvreEtAl2024,
  title = {MultiSim},
  author = {Lepauvre, Alex and Chu, Qian and Zeidman, Peter and Melloni, Lucia},
  year = {2025},
  doi = {https://doi.org/10.5281/zenodo.17231750},
}
```
