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

## Generative model

For each subject $s$, we simulate epoched data $\mathbf{Y}_s \in \mathbb{R}^{N_{\text{samples}} \times n_{\text{feat}}}$ according to the general linear model:

$$
\mathbf{Y}_s = \mathbf{X} \mathbf{B}_s + \mathbf{1} \bm{\beta}_{0,s}^\top + \bm{\varepsilon}_s,
$$

where: 

- $\mathbf{X}$ is the full design matrix with $n_{samples} * n_{trials}$ rows and $n_{samples} * n_{conditions}$ columns
- $\mathbf{B}$ is a matrix with $n_{samples} * n_{conditions}$ rows and $n_{features}$ columns 
- $N_{\text{samples}} = n_{\text{epochs}} \times n_t$ is the total number of time samples across all trials, and $\mathbf{1} \bm{\beta}_{0,s}^\top$ is a subject-specific intercept term. The matrix $\mathbf{X}$ is the **full design matrix**, with one regressor for each combination of experimental condition and time point.

The noise term is drawn from a multivariate normal distribution:

$
\bm{\varepsilon}_s \sim \mathcal{N}(0, \sigma^2 \mathbf{\Sigma}),
$

where $\sigma = \texttt{noise\_std}$ and $\mathbf{\Sigma} = \texttt{spat\_cov}$ denotes the spatial covariance of the sensors (default: identity).

## Constructing the regression coefficients

Each experimental effect is defined by:

- an experimental condition (column index $c$),
- a temporal window $[t_{\text{on}}, t_{\text{off}}]$,
- and a desired multivariate effect size $d$ (interpreted as a Cohen-style $d'$).

We first create a rectangular temporal mask:

$$
m_t = \begin{cases}
1 & \text{if } t_{\text{on}} \leq t \leq t_{\text{off}}, \\
0 & \text{otherwise}
\end{cases}
$$

If a causal kernel $h$ is provided, the temporal profile is convolved and rescaled to unit energy:

$$
\widetilde{\mathbf{m}} = \frac{\mathbf{m} * h}{\lVert \mathbf{m} * h \rVert_2}
$$

A uniform spatial pattern is used: $\mathbf{v} = \mathbf{1} \in \mathbb{R}^{n_{\text{feat}}}$. Its length under the inverse spatial covariance is:

$$
L = \sqrt{\mathbf{v}^\top \mathbf{\Sigma}^{-1} \mathbf{v}} = \sqrt{\operatorname{tr}(\mathbf{\Sigma}^{-1})}
$$

We then scale the amplitude for subject $s$ as:

$$
a_s = \mathcal{N}(d \cdot \sigma / L,\; \texttt{intersub\_noise\_std}^2)
$$

Finally, the corresponding rows in the coefficient matrix $\mathbf{B}_s$ are populated as:

$$
\beta_{c,t:s} = a_s \cdot \widetilde{m}_t \cdot \mathbf{v}^\top
$$

All other entries of $\mathbf{B}_s$ remain zero.

---

## From effect size to decoding accuracy

Because $\lVert \widetilde{\mathbf{m}} \rVert_2 = 1$, the effective Mahalanobis distance between classes at each time point is:

$$
d'_t = \frac{a_s}{\sigma} \cdot L = d
$$

This guarantees that the discriminability between class centroids at each time point matches the desired $d'$. Under standard assumptions of equal-covariance Gaussian classes, the theoretical decoding accuracy is:

$$
P_{\text{correct}} = \Phi\left(\frac{d}{2}\right)
$$

Thus, users can simulate data with known decoding difficulty:  
e.g. $d = 0.2, 0.5, 1.0$ yields ~60%, 69%, and 84% expected accuracy respectively, independent of the number or covariance of features.

# Code Quality and Documentation
SimMEG is hosted on GitHub. Examples and API documentation are available on the platform XXX. We provide installation guides, algorithm introductions, and examples of using the package with Jupyter Notebook [REF]. The package is available on Linux, macOS and Windows for Python >=3.12
It can be installed with pip install simMEG. To ensure high code quality, all implementations adhere to the PEP8 code style [REF], enforced by ruff [REF], the code formatter black and the static analyzer prospector. The documentation is provided through docstrings using the NumPy conventions and build using Sphinx. 

# Acknowledgements

# References
