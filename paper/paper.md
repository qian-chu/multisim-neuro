---
title: 'MultiSim: A Python Toolbox for Simulating Datasets with time resolved multivariate effects
tags:
  - Python
  - neuroscience
  - EEG
  - MEG
  - iEEG
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
Multivariate pattern analysis (MVPA) is now routine in cognitive neuroscience for probing how the brain represents information {cite}`ritchie2019decoding;haynes2006decoding;kriegeskorte2008representational;haxby2001distributed;poldrack2009decoding`. Applied to high-temporal-resolution electrophysiology signals such as electro and magneto-encephalography (EEG and MEG respectively), decoding techniques reveal the millisecond-by-millisecond unfolding of mental representations {cite}`cichy2014resolving;king2014characterizing;king2016brain;cogitate2025adversarial;kok2017prior`. Strinkingly, despite the ubiquity of MVPA techniques, to our knowledge, not method exists to tests the sensitivity and specifity of decoding analysis pipelines, nor to estimate, before data collection, how many trials and how many participants are required to detect an effect of a given size

MultiSim addresses this gap by letting investigators simulate time-resolved multi-channel signals with paramaters matching that of their recording setups, and specify multivariate effects with known timing, spatialization and strength, while controlling channel covariance, sensory noise and between subjects variability. Our algorithm produces multi-subject data sets in which ground truth effects are known. By running their pipeline on these data, researchers obtain a direct read-out of its true-positive rate (can it recover the injected effects?) and false-positive rate (does it raise alarms when nothing is present). In addition, our simulator can be used to perform computational power analysis, to determine the number of trials and subjects, by iterating over these parameters.

This toolbox promotes best-practice MVPA by giving researchers a tailored benchmark for their specific experimental designs, a testbed for developing new decoding methods, and a principled way to check that planned studies are properly powered—ultimately enabling more reliable and efficient investigations of brain function.


# Code Quality and Documentation
SimMEG is hosted on GitHub. Examples and API documentation are available on the platform [here](https://alexlepauvre.github.io/meeg_simulator/). We provide installation guides, algorithm introductions, and examples of using the package with [Jupyter Notebook](https://alexlepauvre.github.io/meeg_simulator/tutorial/index.html). We further provide the full mathetmatical details of our simulation [here](https://alexlepauvre.github.io/meeg_simulator/tutorial/06-mathematical_details.html). The package is available on Linux, macOS and Windows for Python >=3.12
It can be installed with pip install simMEG. To ensure high code quality, all implementations adhere to the PEP8 code style [REF], enforced by ruff [REF], the code formatter black and the static analyzer prospector. The documentation is provided through docstrings using the NumPy conventions and build using Sphinx. 

# Acknowledgements

# References


# Supplementary

## Effect size formulation

Effect sizes are simulated based on the Mahalanobis distance {cite}`mclachlan1999mahalanobis;mahalanobis1930tests`, which is a multivariate generalization of the standard z-score, taking into account the covariance structure of the data. For two classes, the Mahalanobis distance is defined as:

$$\Delta =  \sqrt{(\mu_{1} - \mu_{2})^{T}\Sigma^{-1}(\mu_{1} - \mu_{2})}$$

where $\mu_{1}$ and $\mu_{2}$ are the centroids of each condition, $\Sigma$ is the covariance matrix and $T$ denotes matrix transpose. In our specific case, as the additive noise is multiplied by the covariance matrix to generate the final data, the effect size is equal to:

$$\Delta =  \sqrt{(\mu_{1} - \mu_{2})^{T}(\sigma^{2}\Sigma)^{-1}(\mu_{1} - \mu_{2})}$$

Which simplifies to:

$$\Delta =  \frac{1}{\sigma}\sqrt{(\mu_{1} - \mu_{2})^{T}\Sigma^{-1}(\mu_{1} - \mu_{2})}$$

To simulate data with the require effect size $\Delta$, we generate a random vector $\tilde v$ by drawing a random number from a standard normal distribution for each channel (which are used as the $\Beta$ of our generative GLM). We then normalize that vector to have a Mahalanobis length of 1:

$$ v = \frac{\tilde v}{\sqrt{(\tilde v)^{T}\Sigma^{-1}(\tilde v)}}$$

The unit vector can be scaled by a constant $a$, such that when the centroids of each conditions are placed on each side of itm the distance between them matches the desired Mahalanobis distance:

$$\Delta =  \frac{1}{\sigma} \sqrt{av^{T}\Sigma^{-1}av}$$

Which simplifies to:

$$\Delta =  \frac{a}{\sigma} \sqrt{v^{T}\Sigma^{-1}v}$$

Where $v$ is a random vector of Mahalanobis length of 1 (i.e. Mahalanobis unit length vector) and $a$ is a constant to scale the effect up or down to achieve the desired effect size. As $v$ is of unit length, the term $\sqrt{v^{T}\Sigma^{-1}v}$ is equal to 1 and the equation simplifies to:

$$\Delta =  \frac{a}{\sigma}$$

To generate a multivariate pattern of the desired effect size, we have to resolve $a$ for $||av||_{\Sigma^{-1}}$ and $sigma$, which gives:

$$a = \Delta * \sigma$$

where: 
- $\Delta =$``effect_size``
- $\sigma$=``noise_std``

By multplying our vector $v$ by the constant $a$, we ensure that the distance between the two classes matches the desired effect size.

### Effect size and decoding accuracy

With equal class covariances, a Bayes-optimal linear classifier achieves:

$$\Phi(-\frac{1}{2}d')$$

Where $\Phi$ is the normal distribution cummulative distribution function {cite}`mclachlan1999mahalanobis;mclachlan2005discriminant`. Accordingly, the maximal theoretical decoding accuracy is equal to:

$$1 - \Phi(-\frac{1}{2}d')$$

Thus an effect size of $d'=0.5$ implies a theoretical ceiling of ≈ 69 % accuracy, $d'=1$ gives ≈ 84 %, and so on.  By scaling the injected pattern according to the formula above, **multisim** ensures that simulated data respect this relationship irrespective of the number of channels or their covariance.