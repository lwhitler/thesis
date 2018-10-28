import numpy as np
import copy
import hera_pspec as hp
from pyuvdata import UVData
from hera_qm.metrics_io import process_ex_ants


def bootstrap_median(uvp, blpairs=None, niters=1000):
    """
    Calculate the error on the median power spectrum by bootstrapping.

    Parameters
    ----------
    uvp : UVPSpec object
        UVPSpec object containing the power spectra
    blpairs : array-like
        Baseline pairs to sample (default is all of the baseline pairs in the
        UVPSpec object)
    niters : int, optional
        Number of resamples to take (default is 1000)

    Returns
    -------
    med_sd : array
        Standard deviation of the bootstrapped medians
    """
    # Default to the baseline pairs in the UVPspec object
    if blpairs is None:
        blpairs = uvp.blpair_array

    med_boots = []
    for i in range(niters):
        sample_blps = sample_blpairs(blpairs)  # Resample with replacement
        med = calc_median(uvp, blpairs=sample_blps)
        med_boots.append(med)
    med_boots = np.asarray(med_boots)
    med_sd = np.std(med_boots, axis=0)
    return med_sd


def calc_21cm_obs_freq(z):
    """
    Calculate the observed frequency of the 21.1 cm line from a redshift.

    Parameters
    ----------
    z : int or float
        Redshift at which to calculate the observed frequency

    Returns
    -------
    f_obs : float
         The observed frequency at the given redshift in MHz
    """
    f_obs = (299792458.0/0.211) / ((1+z)*1e6)
    return f_obs


def calc_flagged_bl_percent(uvd, bls):
    """
    Calculate the percentage of flagged baselines.

    Parameters
    ----------
    uvd : UVData object
        UVData object containing the flag array
    bls : array-like
        Baselines for which to look for flags

    Returns
    -------
    flag_percent : array
        The percentage of flagged baselines for each frequency and time with
        dimension (Ntimes, Nfreqs)
    """
    # Get the flags for each of the baselines
    bl_flag_arr = []
    for bl in bls:
        ant1, ant2 = bl[0], bl[1]
        # Not using the get_flags function in hera_pspec because it's slow
        bl_ind = np.where((uvd.ant_1_array == ant1) &
                          (uvd.ant_2_array == ant2))[0]
        bl_flags = uvd.flag_array[bl_ind][:, 0, :, 0].astype(float)
        bl_flag_arr.append(bl_flags)
    bl_flag_arr = np.asarray(bl_flag_arr)

    # Calculate the percentage of baselines flagged
    flag_sum = np.sum(bl_flag_arr, axis=0)
    flag_percent = flag_sum / len(bls)
    return flag_percent


def calc_median(uvp, blpairs=None):
    """
    Find the median of power spectra along the baseline axis.

    Parameters
    ----------
    uvp : UVPSpec object
        UVPSpec object containing the power spectra to take the median of
    blpairs : list or array, optional
        Baseline pairs to take the median over (default is all of the baseline
        pairs in the UVPSpec object)

    Returns
    -------
    median : MaskedArray
         The median along the baseline axis
    """
    # Default to the baseline pairs in the UVPSpec object and take the median
    if blpairs is None:
        ps_data = np.real(uvp.data_array[0][:, :, 0])
        median = np.ma.median(ps_data, axis=0)

    # If baseline pairs are given, stack their data and take the median
    else:
        ps_data = []
        for blp in blpairs:
            # Not using the get_data function in hera_pspec because it's slow
            blp_ind = np.where(uvp.blpair_array == blp)[0]
            uvp_data = uvp.data_array[0][blp_ind, :, 0]
            ps_data.append(np.real(uvp_data))
        ps_data = np.asarray(ps_data)
        median = np.ma.median(ps_data[:, 0, :], axis=0)
    return median


def chan_to_freqs(chans):
    """
    Convert channel numbers to frequencies.

    Parameters
    ----------
    chan : array
        Channels to convert

    Returns
    -------
    freqs : array
         Frequencies in MHz corresponding to the input channels
    """
    freq_per_chan = 100. / 1024.  # 1024 channels over 100 MHz bandwidth
    freqs = 100. + freq_per_chan*chans
    return freqs


def find_good_bls(bls, xants):
    """
    Find good baselines by removing baselines with bad antennas.

    Parameters
    ----------
    bls : array-like
        Initial set of baselines
    xants : array-like
        Bad antennas

    Returns
    -------
    good_bls : list
         The baselines that do not have bad antennas
    """
    for xant in xants:
        # Find and remove baselines with bad antennas
        bad_bls = [bad_bl for bad_bl in bls if xant in bad_bl]
        good_bls = [bl for bl in bls if bl not in bad_bls]
    return good_bls


def sample_blpairs(blpairs, size=None, seed=None):
    """
    Randomly sample a set of baseline pairs with replacement.

    Parameters
    ----------
    blpairs : list or array
        Baseline pairs to sample from
    size : int, optional
        The sample size to return (default is the same size as the input
        baseline pair sample)
    seed : int, optional
        Seed for random number generator (default is no seed)

    Returns
    -------
    sample : list
         The random sample of baseline pairs
    """
    if size is None:
        size = len(blpairs)
    if seed is not None:
        np.random.seed(seed)
    sample = [random.choice(blpairs) for i in range(size)]
    return sample
