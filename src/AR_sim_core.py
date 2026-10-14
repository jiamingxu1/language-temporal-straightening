import numpy as np
import statsmodels.api as sm
import os 

def make_signal_innov(n_voxs, nT, signal_innov_std, corr, beta=0.5):
    """
    n_voxs, nT:            number of voxels and time points
    signal_innov_std:      signal innovation std (should be the total std)
    corr (alpha):          fraction of innovation variance that is shared across voxels
    beta:                  fraction of shared variance going to shared_signal_inno
                           (rest 1-beta goes to shared_noise)
    Returns
    -------
    private_signal_innov  : (n_voxs, nT)
    shared_signal_innov    : (nT,)
    shared_noise_innov     : (nT,)
    """

    sigma = signal_innov_std  # just renaming

    # Variance allocations
    var_private        = (1.0 - corr) * sigma**2
    var_shared_signal  = corr * beta * sigma**2
    var_shared_noise   = corr * (1.0 - beta) * sigma**2

    # Corresponding stds
    std_private        = np.sqrt(var_private)
    std_shared_signal  = np.sqrt(var_shared_signal)
    std_shared_noise   = np.sqrt(var_shared_noise)

    # Draw components
    private_signal_innov      = np.random.randn(n_voxs, nT) * std_private          # voxel-specific
    shared_signal_innov       = np.random.randn(nT) * std_shared_signal            # shared across voxels & runs
    shared_noise_innov        = np.random.randn(nT) * std_shared_noise             # shared across voxels, per run
    # shared_noise_innov        = np.random.randn(n_voxs, nT) * std_shared_noise     # NOT shared across voxels, per run

    return private_signal_innov, shared_signal_innov, shared_noise_innov

# simulate noise-less AR processes
def make_noiseless_arp(ar_coeff, nT, n_voxs, private_signal_innov, shared_signal_innov):
    time_series = np.zeros((n_voxs, nT))
    time_series[:,0] = private_signal_innov[:,0]
    for t in range(1, nT):
        time_series[:, t] = ar_coeff * time_series[:, t - 1] + private_signal_innov[:,t] + shared_signal_innov[t]
        
    return time_series #(n_voxs, nT)


# simulate noisy AR processes 
def make_additive_noisy_arp(ar_coeff, nT, n_voxs, signal_innov, measurement_noise_std):
    white_noise = np.random.randn(n_voxs, nT) * measurement_noise_std

    time_series = np.zeros((n_voxs, nT))
    time_series[:,0] = signal_innov[:,0]

    for t in range(1, nT):
        time_series[:, t] = ar_coeff * time_series[:, t - 1] + signal_innov[:,t]


    noisy_time_series = time_series + white_noise

    return noisy_time_series


zs = lambda v: (v-v.mean(0))/v.std(0)