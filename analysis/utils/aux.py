import numpy as np
import copy
import os
import hera_pspec as hp
from pyuvdata import UVData
from hera_qm.metrics_io import process_ex_ants
import params


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
    f_obs = f_HI * 1e-6 / (1+z)
    return f_obs


def calc_flagged_bl_percent(uvd, bls):
    """
    Calculate the percentage of flagged baselines.

    Parameters
    ----------
    uvd : UVData object
        UVData object containing the flag array
    bls : list of tuples
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
        # Not using the get_flags function in pyuvdata because it's slow
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
    freq_per_chan = (params.max_freq-params.min_freq) / params.Nchans  # MHz/channel
    freqs = params.min_freq + freq_per_chan*chans
    return freqs


def compare_flag_strategies(uvd1, uvd2, bls):
    """
    Comparison of flagging strategies.

    Parameters
    ----------
    uvd1, uvd2 : UVData objects
        UVData objects containing the flag arrays
    bls : tuple
        The baselines for which to compare flags

    Returns
    -------
    flag_comparison : array
        A single array of dimension (Ntimes, Nfreqs) where 0 corresponds to
        no flags, 1 corresponds to being flagged by both methods, 2 corresponds
        to being flagged in uvd1 and not uvd2, and 3 corresponds to being
        flagged in uvd2 and not uvd1
    """
    flags1 = calc_flagged_bl_percent(uvd1, bls)
    flags2 = calc_flagged_bl_percent(uvd2, bls)
    flags1, flags2 = (flags1 == 1), (flags2 == 1)

    flag_comparison = np.zeros_like(flags1, dtype=float)
    flag_comparison[~flags1 & ~flags2] = 0 # Not flagged
    flag_comparison[flags1 & flags2] = 1 # Flagged in both uvd1 and uvd2
    flag_comparison[flags1 & ~flags2] = 2 # Flagged in uvd1 and not uvd2
    flag_comparison[~flags1 & flags2] = 3 # Flagged in uvd2 and not uvd1
    return flag_comparison

def convert_Jy_to_mK(uvd, psbeam):
    """
    Convert data from Jy to mK.

    Parameters
    ----------
    uvd : UVData object
        UVData object containing the data to be converted
    psbeam : PSpecBeam object
        PSpecBeam object to use to convert the data
    """
    freq_array = np.unique(uvd.freq_array)
    uvd.data_array *= psbeam.Jy_to_mK(freq_array, pol='xx')[None, None, :, None]
    uvd.vis_units = 'mK'


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


def freq_to_chans(freqs):
    """
    Convert frequencies to channel numbers.

    Parameters
    ----------
    chan : array
        Channels to convert

    Returns
    -------
    freqs : array
         Frequencies in MHz corresponding to the input channels
    """
    freq_per_chan = (params.max_freq-params.min_freq) / params.Nchans  # MHz/channel
    chans = (freqs-params.min_freq) / freq_per_chan
    return chans


def get_uvpspec(uvd, psbeam, bls1, bls2, spw, uvp_file, time_thresh=0.2):
    """
    Make or load the UVPSpec object corresponding to the data given.

    Parameters
    ----------
    uvd : UVData object
        UVData object containing the data
    psbeam : PSpecBeam object
        PSpecBeam object containing the beam
    bls1, bls2 : array-like
        The baselines to use in OQE
    spw : list of tuples
        Which spectral window to use
    uvp_file : str
        The path to look for/write the UVPSpec object
    time_thresh : float, optional
        Time threshold for broadcasting flags (default is 0.2, not recommended
        to make this larger than 0.5 as per hera_pspec)

    Returns
    -------
    ds : PSpecData object
        The PSpecData object containing the UVData objects with broadcasted
        flags
    uvp : UVPSpec object
        The UVPSpec object containing the power spectra
    """
    ds = hp.pspecdata.PSpecData([uvd, uvd], wgts=[None, None], beam=psbeam)
    ds.broadcast_dset_flags(time_thresh=time_thresh)
    if os.path.isfile(uvp_file):
        # Load UVPSpec from existing file
        print(uvp_file + ' exists, loading UVPSpec...')
        uvp = hp.UVPSpec()
        uvp.read_hdf5(uvp_file)
    else:
        # Run OQE with identity weighting and I normalization
        uvp = ds.pspec(bls1, bls2, dsets=(0, 1), pols=[('XX', 'XX')],
                       spw_ranges=spw, input_data_weight='identity', norm='I',
                       taper='blackman-harris')
        print('\nWriting ' + uvp_file + '...')
        ps_directory = uvp_file.split('tt' + str(time_thresh))[0] + 'tt' + str(time_thresh)
        if not os.path.exists(ps_directory):
            os.makedirs(ps_directory)
        uvp.write_hdf5(uvp_file)
    return ds, uvp


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
    sample = [np.random.choice(blpairs) for i in range(size)]
    return sample


def subtract_medians(median1, median2, med_err1, med_err2):
    """
    Subtract two medians and add their errors in quadrature.

    Parameters
    ----------
    median1, median2 : array
        The medians to subtract (median1 - median2)
    med_err1, med_err2 : array
        The errors on the subtracted medians

    Returns
    -------
    med_diff : array
        The difference of the medians
    med_diff_err : array
        The error of the differenced median
    """
    med_diff = median1 - median2
    med_diff_err = np.sqrt(med_err1**2 + med_err2**2)
    return med_diff, med_diff_err
