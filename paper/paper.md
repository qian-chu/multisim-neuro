# SimMEG: A Python Toolbox for Simulating MEG/EEG Datasets with Known “Ground Truth” Effects

# Summary
In magnetoencephalography (MEG) and electroencephalography (EEG) research, it is often challenging to verify whether the analysis pipelines used to detect neural effects are both sensitive (i.e., detecting real effects) and specific (i.e., avoiding false alarms). Typically, real experimental datasets do not include known ground-truth patterns. This makes it difficult to validate new methods or confirm that existing pipelines are reliably capturing genuine neural signals. SimMEG addresses this gap by allowing researchers to generate realistic MEG/EEG data, complete with user-specified and time-locked experimental effects of known magnitude. The simulated datasets can then be analyzed with any chosen pipeline (e.g., classification, univariate sensor-level analysis, source localization), providing a ground-truth benchmark for sensitivity and specificity.
This toolbox is designed to flexibly mimic user-specified designs and to inject multivariate effects in chosen time windows, reflecting hypothesized differences between experimental conditions. By doing so, it aids in:
- Confirming whether a pipeline can detect signals of a given effect size.
- Demonstrating to reviewers and collaborators that analyses are robust, reproducible, and well-calibrated.
- Teaching and training newcomers to MEG/EEG analysis in a controlled environment.

In the following sections, we outline the general rationale for the toolbox (Introduction), detail how our simulation function generates data (Methods), provide a simple demonstration (Results), and discuss potential applications and future developments (Discussion).

# Statement of needs
MEG and EEG are popular techniques for studying brain dynamics at the millisecond timescale. The analysis of these data, however, can be complex due to high dimensionality, low signal-to-noise ratios, and subject-to-subject variability. Researchers routinely develop pipelines that involve preprocessing (artifact removal, filtering), sensor- or source-level analyses (e.g., multivariate decoding, univariate tests), and statistics (e.g., cluster-based permutation tests).
One major limitation in evaluating such pipelines is that real datasets rarely provide absolute certainty about the presence, timing, or amplitude of neural effects. Often, “chance-level” or baseline-corrected activity is used as a reference, but the ground truth remains unknown. SimMEG directly addresses this limitation:
1.	Design-driven simulation: Users specify an experimental design matrix, including factors (e.g., face vs. object stimuli; attended vs. unattended conditions).
2.	Time-locked pattern injection: The toolbox injects effects in specified time windows so that the timing and amplitude of each effect are known.
3.	Multi-subject support: Multiple participants can be simulated for group-level analyses.
4.	Spatial covariance: Researchers can incorporate realistic sensor covariance or keep it identity for simplicity.
By running a chosen analysis pipeline on these artificial datasets, researchers can easily verify whether and where the pipeline correctly detects the known effects, thereby obtaining a direct estimate of sensitivity and specificity.

# Method

SimMEG generates synthetic epoched MEG/EEG data according to a user-specified design matrix and effect windows. We briefly describe the mathematical model and outline each step of the procedure.
1.	Design Matrix
Let $X\mathbf{X}$ be a design matrix of size $[N_{\text{trials}}, N_{\text{cond}}]$, indicating how each trial corresponds to one or more experimental factors (e.g., Category, Attention). Each column in $X\mathbf{X}X$ encodes a condition contrast (e.g., +1+1+1 for faces, −1-1−1 for objects).
2.	Time Grid and Activation Windows
We simulate each epoch from $t_{\min}$ to $t_{\max}$, sampled at frequency $f_s$. If $T$ is the total number of time points in each epoch, we construct an identity matrix $\mathbf{X}_t \in \mathbb{R}^{T \times T}$ so that each time point can be considered separately in the design. Next, we specify effect windows for each condition, e.g., faces vs. objects is active from 100 ms to 200 ms. A binary activation vector $\mathbf{cv}$ indicates which time samples are “activated” for each condition.
3.	Kronecker Product for $Trial \times Time$
We define the full “trial-by-time” design matrix $\mathbf{X}_{\text{full}}$ via the Kronecker product:
$\mathbf{X}_{\text{full}} = \mathbf{X} \otimes \mathbf{X}_t$,
ensuring that each column in $\mathbf{X}_{\text{full}}$ corresponds to a unique combination of (condition, time).
4.	Generating the Ground Truth Effects
We assume each condition contrast has a latent multivariate pattern $\boldsymbol{\beta}$. We can sample $\boldsymbol{\beta}$ from a multivariate normal distribution, incorporating a user-specified spatial covariance $\mathbf{\Sigma} \in \mathbb{R}^{C \times C}$ for the $C$ channels (or sensor “modes”). The resulting parameter matrix $\mathbf{B}$ has shape $[T \times N_{\text{cond}},\, C]$. Crucially, we only activate the relevant columns (time windows) for each condition by diagonalizing the time-activation mask $\mathbf{cv}$.
5.	Adding an Intercept and Noise
We introduce an intercept term $\mathbf{B}_0$ (sampled from a normal distribution) and add random Gaussian noise $\varepsilon$ with standard deviation $\sigma$. The noise can also be correlated across channels:
$\boldsymbol{\varepsilon} \sim \mathcal{N}\left(\mathbf{0},\, \sigma^2 \mathbf{\Sigma}\right)$.
The final simulated data $\mathbf{Y}$ in vectorized form is:
$\mathbf{Y}_{\text{vec}} = \mathbf{X}_{\text{full}} \mathbf{B} \;+\; \mathbf{X}_0 \mathbf{B}_0 \;+\; \boldsymbol{\varepsilon}$.
Here, $\mathbf{X}_0$ can be an all-ones column (intercept). We then reshape $\mathbf{Y}$ back to the shape $[\;N_{\text{trials}},\, T,\, C\;]$.
6.	Epochs Construction
The final data is placed into MNE-Python EpochsArray objects (one per simulated subject), making it seamlessly compatible with typical pipelines for preprocessing, decoding, or sensor-level analysis.

# Code Quality and Documentation
SimMEG is hosted on GitHub. Examples and API documentation are available on the platform XXX. We provide installation guides, algorithm introductions, and examples of using the package with Jupyter Notebook [REF]. The package is available on Linux, macOS and Windows for Python >=3.12
It can be installed with pip install simMEG. To ensure high code quality, all implementations adhere to the PEP8 code style [REF], enforced by ruff [REF], the code formatter black and the static analyzer prospector. The documentation is provided through docstrings using the NumPy conventions and build using Sphinx. 

# Acknowledgements

# References
