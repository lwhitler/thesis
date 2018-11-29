import numpy as np
import copy
from pyuvdata import UVData
from pyuvdata import utils as uvutils
from hera_sim import foregrounds, noise, rfi, sigchain


def bl_len_in_ns(len_array):
    """
    Compute the length of a baseline in nanoseconds.

    Parameters
    ----------
    len_array : array of scalars, lists, or tuples
        Array of baseline lengths or coordinate-wise lengths
    """
    len_in_ns = (np.linalg.norm(len_array) / 299792458.) * 1e9
    return len_in_ns


def simulate_one_data_file(data_in, file_type, sim_out=None, clobber=False,
                           Trx=150., broadband_rfi=0.01, random_rfi=0.001,
                           nsrcs=200, xtalk_amp=3.):
    """
    Simulate a data file using an existing file as a base.

    Parameters
    ----------
    data_in : string
        The data to use as a starting point
    file_type : one of ['uvfits', 'miriad', 'uvh5']
        The input and output file type
    sim_out : string, optional
        The file to write the simulation to (default is '.sim' appended
        to the end of the input data path)
    clobber : bool, optional
        Option to overwrite the output file if it already exists
        (default is false)
    Trx : float, optional
        Receiver temperature (default 150 K)
    broadband_rfi : float
        Chance of broadband RFI occurring
    random_rfi : float
        Chance of random RFI occurring
    nsrcs : int
        Number of point sources to simulate
    xtalk_amp : float
        Amplitude of crosstalk
    """
    print('Loading data...')
    # Load the real data to start from
    uvd = UVData()
    uvd.read(data_in, file_type=file_type)
    uvd_sim = copy.deepcopy(uvd)  # To save the simulation
    # Frequencies, LSTs, antennas, baselines, and polarizations in the real data
    freqs = np.unique(uvd.freq_array) / 1e9  # In GHz for hera_sim
    lst_inds = np.unique(uvd.lst_array, return_index=True)[1]
    lsts = np.asarray([uvd.lst_array[i] for i in sorted(lst_inds)])  # To preserve order
    bls = np.unique(uvd.baseline_array)
    pols = uvutils.polnum2str(uvd.polarization_array)
    antpols1 = [(ant, pol[0]) for ant in set(uvd.ant_1_array) for pol in pols]
    antpols2 = [(ant, pol[1]) for ant in set(uvd.ant_1_array) for pol in pols]
    antpols = list(set(antpols1).union(set(antpols2)))

    print('Finding redundant baselnes...')
    # Redundant baseline groups in the real data
    unique_bls, bl_inds = np.unique(uvd.baseline_array, return_index=True)
    reds, _, lens, conj_bls = uvutils.get_baseline_redundancies(unique_bls, uvd.uvw_array[bl_inds],
                                                                tol=0.5, with_conjugates=True)
    # Throw out autos
    auto_ind = np.where(lens == 0)
    del reds[auto_ind[0][0]]
    lens = np.delete(lens, auto_ind)

    ### BUILDING THE MODEL ###

    # Sky and receiver temperature
    Tsky_model = {pol: noise.HERA_Tsky_mdl[pol] for pol in pols}
    Tsky = {pol: noise.resample_Tsky(freqs, lsts, Tsky_mdl=Tsky_model[pol]) for pol in pols}
    Trx = Trx

    # Antenna gains
    print('Simulating antenna gains...')
    gains = sigchain.gen_gains(freqs, antpols)

    # RFI
    print('Simulating RFI...')
    rfi_narrow = rfi.rfi_stations(freqs, lsts)
    rfi_broad = rfi.rfi_impulse(freqs, lsts, chance=broadband_rfi)
    rfi_scatter = rfi.rfi_scatter(freqs, lsts, chance=random_rfi)
    rfi_all = rfi_narrow + rfi_broad + rfi_scatter

    print('Simulating foregrounds...')
    # Combine foregrounds, noise, RFI, and antenna gains
    bl_dict = {bl: np.zeros((len(lsts), len(freqs), len(pols))) for bl in bls}
    for red_group, bl_len in zip(reds, lens):
        bl_len = bl_len_in_ns(bl_len)
        pt_src = foregrounds.pntsrc_foreground(lsts, freqs, bl_len, nsrcs=nsrcs)
        for i, pol in enumerate(pols):
            diffuse = foregrounds.diffuse_foreground(Tsky_model[pol], lsts, freqs, bl_len)
            true_vis = diffuse + pt_src
            for bl in red_group:
                # Conjugate the visibilities if baseline is conjugately redundant
                if bl in conj_bls:
                    true_vis_bl = true_vis.conj()
                else:
                    true_vis_bl = true_vis
                # Add noise and crosstalk for each baseline
                noise_sim = noise.sky_noise_jy(Tsky[pol] + Trx, freqs, lsts)
                xtalk = sigchain.gen_xtalk(freqs, amplitude=xtalk_amp)
                vis = true_vis_bl + noise_sim + xtalk
                # Apply antenna gains and save to dictionary
                bl_tuple = uvd.baseline_to_antnums(bl)
                g_ij = gains[(bl_tuple[0], pol[0])] * gains[(bl_tuple[1], pol[1])].conj()
                bl_dict[bl][:, :, i] = vis * g_ij

    # Fit the model into the UVData structure
    sim_data = np.zeros_like(uvd.data_array)
    for bl in bl_dict.keys():
        blt_ind = np.where(uvd.baseline_array == bl)[0]
        sim_data[blt_ind] = bl_dict[bl][:, None, :, :]

    # Write the simulation out
    if sim_out is None:
        sim_out = data_in + '.sim'
    print('Saving to {0}...'.format(sim_out))
    # Unflag everything just in case anything was flagged
    uvd_sim.flag_array = np.full_like(uvd.flag_array, False)
    uvd_sim.data_array = sim_data
    if file_type == 'miriad':
        uvd_sim.write_miriad(sim_out)
    elif file_type == 'uvfits':
        uvd_sim.write_uvfits(sim_out)
    elif file_type == 'uvh5':
        uvd_sim.write_uvh5(sim_out)
