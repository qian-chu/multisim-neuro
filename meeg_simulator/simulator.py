import numpy as np
import mne
from typing import List, Optional


def simulate_data(
    X: np.ndarray,
    noise_std: float,
    n_modes: int,
    n_subjects: int,
    tmin: float,
    tmax: float,
    sfreq: float,
    t_win: np.ndarray,
    effects: np.ndarray,
    effects_amp: Optional[np.ndarray] = None,
    spat_cov: Optional[np.ndarray] = None,
    ch_type: Optional[str] = "eeg",
) -> List[np.ndarray]:
    """Simulate epoched MEG/EEG data with multivariate patterns.

    This function generates simulated EEG/MEG data with predefined experimental 
    effects, allowing for controlled evaluation of analysis methods. Effects are 
    introduced at specified time windows, serving as ground truth signals. 
    The data is structured into epochs, making it compatible with `mne.Epochs`.

    The method used here is based on the approach implemented in **SPM's `DEMO_CVA_RSA.m`** 
    function, originally developed by **Karl Friston and Peter Zeidman** [1]_. 
    Our implementation extends their method by incorporating **time-resolved effects**, 
    allowing for dynamic experimental manipulations over specified time windows.

    Parameters
    ----------
    X : array, shape (n_trials, n_experimental_conditions)
        Between-trial design matrix specifying experimental manipulations.
    noise_std : float
        Standard deviation of additive Gaussian noise (applied before spatial covariance).
    n_modes : int
        Number of spatial modes (e.g., sensors or components) in the simulated data.
    n_subjects : int
        Number of subjects to simulate.
    tmin : float
        Start time (in seconds) of each epoch.
    tmax : float
        End time (in seconds) of each epoch.
    sfreq : float
        Sampling frequency in Hz.
    t_win : array, shape (n_effects, 2)
        Time windows (start, end) in seconds where each experimental effect is nonzero.
    effects : array, shape (n_effects,)
        Indices of the experimental conditions (columns of `X`) associated with 
        time-locked effects. Each entry corresponds to a row in `t_win`.
    effects_amp : array, shape (n_effects,) | None, optional
        Amplitudes of the effects. Effects are simulated by sampling beta parameters 
        from a normal distribution across channels, with `effects_amp` defining 
        the variance of the distribution. Default is None (scales are uniform).
    spat_cov : array, shape (n_modes, n_modes) | None, optional
        Spatial covariance matrix for the simulated data. If None, an identity 
        matrix is used (i.e., no cross-channel correlations).
    ch_type : str, optional
        Type of simulated channels, e.g., `'eeg'` or `'meg'`. Default is `'eeg'`.

    Returns
    -------
    epochs_list : list of mne.Epochs
        A list of `mne.Epochs` objects, one per subject (length `n_subjects`).

    Raises
    ------
    ValueError
        If the number of rows in `t_win` does not match the length of `effects`.

    Notes
    -----
    - This function follows the **same methodological principles** as `DEMO_CVA_RSA.m` from SPM, 
      but extends it by adding time-resolved experimental effects.
    - The original implementation in SPM was developed by **Karl Friston and Peter Zeidman**.
    - The generated data follows an event-related structure, suitable for 
      classification and decoding analyses.
    - Effects are injected into selected experimental conditions based on `X`.

    References
    ----------
    .. [1] Friston, K., & Zeidman, P. "DEMO_CVA_RSA.m", Statistical Parametric Mapping (SPM).
           Available at: https://github.com/spm/spm/blob/main/toolbox/DEM/DEMO_CVA_RSA.m

    Examples
    --------
    Simulating a simple dataset with 20 subjects and a single experimental effect:

    >>> import numpy as np
    >>> from meeg_simulator import simulate_data
    >>> X = np.random.randn(100, 2)  # 100 trials, 2 experimental conditions
    >>> t_win = np.array([[0.2, 0.5]])  # Effect between 200-500 ms
    >>> effects = np.array([1])  # Effect corresponds to second column of X
    >>> epochs_list = simulate_data(X, noise_std=0.1, n_modes=64, n_subjects=20,
    ...                             tmin=-0.2, tmax=0.8, sfreq=250, 
    ...                             t_win=t_win, effects=effects)
    >>> print(len(epochs_list))  # Should return 20 subjects

    """
    # ---------------------------------------------------------------------
    # 1. Check inputs & compute the number of samples
    # ---------------------------------------------------------------------
    if len(effects) != len(t_win):
        raise ValueError(
            "The dimension of 'effects' and 't_win' do not match! "
            "There should be exactly one time window for each effect."
        )
    if effects_amp is None:
        effects_amp = [1 / 32] * len(effects)
    n_trials, n_exp_conditions = X.shape
    # Make sure we get an integer sample count:
    n_samples = int(round((tmax - tmin) * sfreq)) - 1
    if n_samples <= 0:
        raise ValueError(
            "Derived 'n_samples' must be positive. Check tmin, tmax, and sfreq."
        )

    # Prepare info for epochs object in the end
    info = mne.create_info(
        [f"CH{n:03}" for n in range(n_modes)], ch_types=[ch_type] * n_modes, sfreq=sfreq
    )

    # ---------------------------------------------------------------------
    # 2. Create a time vector for each epoch and the FIR (identity) within-trial design
    # ---------------------------------------------------------------------
    t = np.linspace(tmin, tmax, n_samples + 1, endpoint=False)[1:]
    # Identity matrix => one column per time point
    Xt = np.eye(n_samples)  # shape => [n_samples, n_samples]

    # ---------------------------------------------------------------------
    # 3. Build a "time activation" matrix for each experimental condition
    #    cv.shape => [n_samples, n_exp_conditions]
    #    This marks time points in which each condition is active (1) or inactive (0)
    # ---------------------------------------------------------------------
    cv = np.zeros((n_samples, n_exp_conditions), dtype=float)

    for idx, eff_cond in enumerate(effects):
        # Identify which samples lie in the desired time window
        start_t, end_t = t_win[idx]
        # Mask for t in [start_t, end_t]
        mask = (t >= start_t) & (t <= end_t)
        cv[mask, eff_cond] = effects_amp[idx]

    # ---------------------------------------------------------------------
    # 4. Construct the full design matrix with the Kronecker product
    #    shape => [n_trials*n_samples, n_exp_conditions*n_samples]
    # ---------------------------------------------------------------------
    X_full = np.kron(X, Xt)

    # ---------------------------------------------------------------------
    # 5. Intercept term across all trials and samples
    #    shape => [n_trials*n_samples, 1]
    # ---------------------------------------------------------------------
    X0 = np.ones((n_trials * n_samples, 1), dtype=float)

    # ---------------------------------------------------------------------
    # 6. Prepare or validate spatial covariance
    #    - If none given, we assume an identity (no cross-mode correlation).
    #    - Must be shape => [n_modes, n_modes] if provided.
    # ---------------------------------------------------------------------
    if spat_cov is None:
        spat_cov = np.eye(n_modes)
    else:
        if spat_cov.shape != (n_modes, n_modes):
            raise ValueError(
                f"spat_cov must be shape ({n_modes}, {n_modes}), "
                f"but got {spat_cov.shape}."
            )

    # ---------------------------------------------------------------------
    # 7. Prepare outputs: loop over subjects and simulate data
    # ---------------------------------------------------------------------
    simulated_data = []

    # Flatten cv to shape [n_samples*n_exp_conditions] so we can build a diagonal "selector"
    # and multiply it by random effects. That ensures only certain (time, condition) combos
    # end up non-zero.
    cv_diagonal = np.diag(
        cv.T.flatten()
    )  # shape => [n_samples*n_exp_conditions, n_samples*n_exp_conditions]

    # Each subject gets unique random draws
    for _ in range(n_subjects):
        # --------------------------------------------------
        # 7a. Build subject-specific effect weights
        #     shape => [n_samples*n_exp_conditions, n_modes]
        # --------------------------------------------------
        # Generate beta parameters randomy sampled from a standard normal distribution,
        # but using CV to set the effects to their desired effect sizes. Adding spatial covariance across effects
        B = (
            cv_diagonal
            @ np.random.randn(n_samples * n_exp_conditions, n_modes)
            @ spat_cov
        )  # shape => [n_samples*n_exp_conditions, n_modes]

        # --------------------------------------------------
        # 7b. Combine with the full design
        #     shape => [n_trials*n_samples, n_modes]
        # --------------------------------------------------
        data = X_full @ B

        # --------------------------------------------------
        # 7c. Add intercept
        #     shape => [n_trials*n_samples, n_modes]
        # --------------------------------------------------
        B0 = np.random.randn(1, n_modes) / 16.0
        intercept = X0 @ B0
        data += intercept

        # --------------------------------------------------
        # 7d. Add noise (spatially correlated)
        # --------------------------------------------------
        noise = np.random.randn(n_trials * n_samples, n_modes)
        # Apply spatial covariance
        noise = noise @ spat_cov
        noise *= noise_std

        data += noise

        # --------------------------------------------------
        # 7e. Store simulated data for this subject
        # --------------------------------------------------
        # Reshape the data:
        # Reshape to [N, T, C] (trials x time x channels)
        data = np.transpose(data.reshape([n_trials, len(t), n_modes]), [0, 2, 1])

        simulated_data.append(mne.EpochsArray(data, info, tmin=tmin, reject_tmin=False, verbose='ERROR'))

    return simulated_data
