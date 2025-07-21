---
title: 'MultiSim: A Python Toolbox for Simulating Datasets with time resolved multivariate effects'
tags:
  - Python
  - neuroscience
  - EEG
  - MEG
  - iEEG
  - LFP
  - simulation
authors:
  - name: Alex Lepauvre
    orcid: 0000-0002-4191-1578
    corresponding: true
    affiliation: "1, 2"
  - name: Qian Chu
    orcid: 0000-0003-2308-6102
    affiliation: "1, 3, 4, 5"
  - name: Lucia Melloni
    orcid: 0000-0001-8743-5071
    affiliation: "1, 6, 7"
  - name: Peter Zeidman
    orcid: 0000-0003-3610-6619
    affiliation: "8"
affiliations:
  - name: Neural Circuits, Consciousness and Cognition Research Group, Max Planck Institute for Empirical Aesthetics, Frankfurt am Main, Germany
    index: 1
    ror: 000rdbk18
  - name: Donders Institute for Brain, Cognition and Behaviour, Radboud University Nijmegen, Nijmegen, The Netherlands
    index: 2
    ror: 016xsfp80
  - name: Max Planck – University of Toronto Centre for Neural Science and Technology
    index: 3
  - name: Krembil Brain Institute, Toronto Western Hospital, University Health Network, Toronto, ON, Canada
    index: 4
    ror: 042xt5161
  - name: Institute of Biomedical Engineering, University of Toronto, Toronto, ON, Canada
    index: 5
    ror: 03dbr7087
  - name: Department of Neurology, New York University Grossman School of Medicine, New York, NY, USA
    index: 6
    ror: 0190ak572
  - name: Predictive Brain Department, Research Center One Health Ruhr, University Alliance Ruhr, Faculty of Psychology, Ruhr University Bochum, Bochum, Germany
    index: 7
    ror: 04tsk2644
  - name: Wellcome Centre for Human Neuroimaging, Institute of Neurology, University College London, London, UK
    index: 8
    ror: 02jx3x895
date: 1 July 2025
bibliography: paper.bib
header-includes:
  - \usepackage{amsmath}
  - \providecommand{\bm}[1]{\boldsymbol{#1}}
---

# Summary

In MEG/EEG research, validating analysis pipelines is hampered by the lack of ground-truth neural signals in real data. SimMEG fills this gap by generating realistic, time-locked multivariate effects of known magnitude that you can inject into simulated sensor data. You can then run any pipeline—e.g. decoding, sensor-level statistics, or source estimation—against these datasets to benchmark sensitivity and specificity.

Key benefits include:

- Testing whether your pipeline reliably detects effects of a chosen size.  
- Providing demonstrable, reproducible benchmarks for reviewers or collaborators.  
- Offering a controlled teaching environment for newcomers.  

Below, we describe the rationale (Statement of needs), and the data-generation method (Methods), a hands-on example (Results), and potential extensions (Discussion).  

# Statement of needs

Multivariate pattern analysis (MVPA) is now routine in cognitive neuroscience for probing how the brain represents information [@ritchie2019decoding;@haynes2006decoding;@kriegeskorte2008representational;@haxby2001distributed;@poldrack2009decoding]. Applied to high-temporal-resolution electrophysiology signals such as electro and magneto-encephalography (EEG and MEG respectively), decoding techniques reveal the millisecond-by-millisecond unfolding of mental representations [@cichy2014resolving;@king2014characterizing;@king2016brain;@cogitate2025adversarial;@kok2017prior]. Strikingly, despite the ubiquity of MVPA techniques, to our knowledge, no method exists to test the sensitivity and specificity of decoding analysis pipelines, nor to estimate, before data collection, how many trials and how many participants are required to detect an effect of a given size.

MultiSim addresses this gap by letting investigators simulate time-resolved multi-channel signals with paramaters matching that of their recording setups, and specify multivariate effects with known timing, spatialization and strength, while controlling channel covariance, sensory noise and between subjects variability. The code block below provides a minimal example, highlighting the simplicity with which multivariate effects can be specified with our toolbox (see \autoref{fig:pipeline}**A** for a visual representation of key parameters):

```python
import numpy as np
from meeg_simulator import simulate_data
X = pd.DataFrame(np.random.randn(100, 1), columns=["face-object"]) # 100 trials, 1 experimental condition
t_win = np.array([[0.2, 0.5]])  # Effect between 200-500 ms
effects = [
    {"condition": 'face-object',
      "windows": [0.1, 0.3],
      "effect_size": 0.5
    }
]
sims = Simulator(
   X, noise_std=0.1, n_channels=64, n_subjects=20,
   tmin=-0.2, tmax=0.8, sfreq=250,
   t_win=t_win, effects=effects
  )
sim.summary()  # Should return 20 subjects
```

Our algorithm produces multi-subject data sets in which ground truth effects are known with precise timiming (see \autoref{fig:pipeline}**B**). Furthermore, our pipelines enable full flexibility regarding the temporal dynamics of the effects (see see \autoref{fig:pipeline}**C**) as well as the temporal generalization of the injected effects (see \autoref{fig:pipeline}**D**), enabling the simulation of all patterns presented by [@king2014characterizing] (Figure 2). By running custom analysis pipeline on simulated data, researchers obtain a direct read-out of its true-positive rate (can it recover the injected effects?) and false-positive rate (does it raise alarms when nothing is present?). In addition, our simulator can be used to perform computational power analysis, to determine the number of trials and subjects, by iterating over these parameters.

![**Figure 1. Overview of the simulation and decoding framework.** **A**. General data parameters for the simulation. Left: `n_channels`  corresponds to the number of channels in the montage, with 2 experimental conditions. Middle: `X` is the design matrix (each column is an experimental condition and each row a trial). Right:  `ch_cov` is the channel by channel covariance matrix of the data to be simulated.  **B**. Minimal simulation example with effects for each experimental condition with large effect size. Left: `effects` is a dictionary that specifies the `"condition"`, time window (`"windows"`) and effect size (`"effect_size"`) of each effect to simulate. The example specifies an effect of category from 0.1 and 0.2 s with an effect size of 4, and an effect of the attention condition from 0.4 to 0.5 with an effect size of 4. Middle: example of the activation of a single channel. Right: resulting decoding accuracy (using a Support vector machine classifier). **C**. Example of simulated effects with a an added gamma kernel to simulate effects with biologically plausible temporal dynamcs. Left: `effects` similar to that of B but with effect size of 0.5 for each condition. Middle: gamma `kernel`, specifying the temporal dynamics of the multivariate effect. Right: Resulting decoding accuracy. **D**. example of simulated data with cross temporal generalization of the category effect. Left: `effects` dictionary specifies two different time windows for the effect of category as a list. Middle: resulting decoding accuracy. Right: Temporal generalization of the decoding.\label{fig:pipeline}](figure1.png){width=100%}

This toolbox promotes best-practice MVPA by giving researchers a tailored benchmark for their specific experimental designs, a testbed for developing new decoding methods, and a principled way to check that planned studies are properly powered—ultimately enabling more reliable and efficient investigations of brain function.

# Code Quality and Documentation

SimMEG is hosted on GitHub. Examples and API documentation are available on the platform [here](https://alexlepauvre.github.io/meeg_simulator/). We provide installation guides, algorithm introductions, and examples of using the package with [Jupyter Notebook](https://alexlepauvre.github.io/meeg_simulator/tutorial/index.html). We further provide the full mathematical details of our simulation [here](https://alexlepauvre.github.io/meeg_simulator/tutorial/06-mathematical_details.html). The package is available on Linux, macOS and Windows for Python >=3.12
It can be installed with pip install simMEG. To ensure high code quality, all implementations adhere to the PEP8 code style [REF], enforced by ruff [REF], the code formatter black and the static analyzer prospector. The documentation is provided through docstrings using the NumPy conventions and build using Sphinx.

# Acknowledgements

# References
```{bibliography}
```

# Supplementary
