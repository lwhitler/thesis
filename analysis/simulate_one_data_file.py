import numpy as np
import copy
from pyuvdata import UVData
from pyuvdata import utils as uvutils
from hera_sim import foregrounds, noise, rfi, sigchain
import utils


# Paths
JD_dec = '40141'
filename = 'zen.2458106.' + JD_dec + '.xx.HH.uv'
data_in = '/data6/HERA/data/IDR2.1/2458106/' + filename
sim_out = '/home/lwhitler/data/dfiles/sim/' + filename.replace('.uv', '.sim.uv')

# Load the real data to start from
uvd = UVData()
uvd.read(data_in, file_type='miriad')
uvd_sim = copy.deepcopy(uvd)  # To save the simulation
# Frequencies, LSTs, antennas, baselines, and polarizations in the real data
freqs = np.unique(uvd.freq_array) / 1e9  # In GHz for hera_sim
lsts, lst_inds = np.unique(uvd.lst_array, return_index=True)
lsts = np.asarray([lsts[i] for i in sorted(lst_inds)])
bls = np.unique(uvd.baseline_array)
pols = uvutils.polnum2str(uvd.polarization_array)
ants = list(set(uvd.ant_1_array).union(set(uvd.ant_2_array)))

# Redundant baseline groups in the real data
unique_bls, bl_inds = np.unique(uvd.baseline_array, return_index=True)
reds, _, lens, conj_bls = uvutils.get_baseline_redundancies(unique_bls, uvd.uvw_array[bl_inds],
                                                            tol=0.5, with_conjugates=True)
# Throw out autos
auto_ind = np.where(lens == 0)
del reds[auto_ind[0][0]]
lens = np.delete(lens, auto_ind)
lens_ns = [utils.aux.bl_len_in_ns(length) for length in lens]

### BUILDING THE MODEL ###
# Sky and receiver temperature
Tsky_model = {pol: noise.HERA_Tsky_mdl[pol] for pol in pols}
Tsky = {pol: noise.resample_Tsky(freqs, lsts, Tsky_mdl=Tsky_model[pol]) for pol in pols}
Trx = 150.

# Antenna gains
gains = sigchain.gen_gains(freqs, ants)

# RFI
rfi_narrow = rfi.rfi_stations(freqs, lsts)
rfi_broad = rfi.rfi_impulse(freqs, lsts, chance=0.01)
rfi_scatter = rfi.rfi_scatter(freqs, lsts, chance=0.001)
rfi_all = rfi_narrow + rfi_broad + rfi_scatter

# Combine foregrounds, noise, RFI, and antenna gains
bl_dict = {bl: np.zeros((len(lsts), len(freqs), len(pols))) for bl in bls}
for i, (red_group, bl_len) in enumerate(zip(reds, lens_ns)):
    pt_src = foregrounds.pntsrc_foreground(lsts, freqs, bl_len, nsrcs=200)
    for j, pol in enumerate(pols):
        diff = foregrounds.diffuse_foreground(Tsky_model[pol], lsts, freqs, bl_len)
        true_vis = diff + pt_src
        for bl in red_group:
            # Conjugate the visibilities if baseline is conjugately redundant
            if bl in conj_bls:
                true_vis_red = true_vis.conj()
            else:
                true_vis_red = true_vis
            # Add noise and crosstalk for each baseline
            noise_sim = noise.sky_noise_jy(Tsky[pol] + Trx, freqs, lsts)
            noisy_vis = true_vis_red + noise_sim + rfi_all
            xtalk = sigchain.gen_xtalk(freqs)
            xtalk_noisy_vis = noisy_vis + xtalk
            # Apply antenna gains and save
            bl_tuple = uvd.baseline_to_antnums(bl)
            g_ij = gains[bl_tuple[0]] * gains[bl_tuple[1]].conj()
            bl_dict[bl][:, :, j] = xtalk_noisy_vis * g_ij

# Set any missed baselines to NaNs (should just be the autos)
print('Setting the following baselines to NaNs:')
for bl in bl_dict.keys():
    if np.count_nonzero(bl_dict[bl]) == 0:
        print('\t{0}'.format(uvd.baseline_to_antnums(bl)))
        bl_dict[bl] = np.full_like(bl_dict[bl], np.nan)

# Fit the model into the UVData structure
sim_data = np.zeros_like(uvd.data_array)
for bl in bl_dict.keys():
    blt_ind = np.where(uvd.baseline_array == bl)[0]
    sim_data[blt_ind] = bl_dict[bl][:, None, :, :]

# Save to a miriad file
# Unflag everything just in case anything was flagged
uvd_sim.flag_array = np.full_like(uvd.flag_array, False)
uvd_sim.data_array = sim_data
uvd_sim.write_miriad(sim_out)
