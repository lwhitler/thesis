import numpy as np
import matplotlib.pyplot as plt
import cmocean
import copy
from scipy import constants
from pyuvdata import UVData
from pyuvdata import utils as uvutils
from hera_sim import foregrounds, noise, rfi, sigchain


def bl_len_in_ns(uvw_arr):
    return (np.linalg.norm(uvw_arr) / 299792458.) * 1e9

# Paths
JD_dec = '40141'
filename = 'zen.2458106.' + JD_dec + '.xx.HH.uv'
data_in = '/data6/HERA/data/IDR2.1/2458106/' + filename
sim_out = '/home/lwhitler/data/dfiles/sim/' + filename.replace('.uv', '.sim.uv')

# Load the real data to start from
uvd = UVData()
uvd.read(data_in, file_type='miriad')
uvd_sim = copy.deepcopy(uvd)  # To save the simulation
# Frequencies, LSTs, and antennas in the real data
freqs = np.unique(uvd.freq_array) / 1e9  # In GHz for hera_sim
lsts = np.unique(uvd.lst_array)
ants = list(set(uvd.ant_1_array).union(set(uvd.ant_2_array)))

# Redundant baseline groups in the real data
unique_bls, bl_inds = np.unique(uvd.baseline_array, return_index=True)
reds, _, lens, conj_bls = uvutils.get_baseline_redundancies(unique_bls, uvd.uvw_array[bl_inds],
                                                            tol=0.5, with_conjugates=True)
# Throw out autos
# NOTE: The autos make hera_sim unhappy, and there may be a better way
# to deal with them than this, but for the sake of sort of getting a result,
# this works.
auto_ind = np.where(lens == 0)
del reds[auto_ind[0][0]]
lens = np.delete(lens, auto_ind)
lens_ns = [bl_len_in_ns(length) for length in lens]

### BUILDING THE MODEL ###
# Sky and receiver temperature
Tsky_model = noise.HERA_Tsky_mdl['xx']
Tsky = noise.resample_Tsky(freqs, lsts, Tsky_mdl=Tsky_model)
Trx = 150.

# Antenna gains
gains = sigchain.gen_gains(freqs, ants)

# RFI
rfi_narrow = rfi.rfi_stations(freqs, lsts)
rfi_broad = rfi.rfi_impulse(freqs, lsts, chance=0.01)
rfi_scatter = rfi.rfi_scatter(freqs, lsts, chance=0.001)
rfi_all = rfi_narrow + rfi_broad + rfi_scatter

# True visibilities for each redundant baseline group
# NOTE: 200 point sources was kind of an arbitrary choice.
true_vis = {}
for i, bl_len in enumerate(lens_ns):
    diff = foregrounds.diffuse_foreground(Tsky_model, lsts, freqs, bl_len)
    pt_src = foregrounds.pntsrc_foreground(lsts, freqs, bl_len, nsrcs=200)
    true_vis[i] = diff + pt_src

# With noise and antenna gains
bl_dict = dict.fromkeys(np.unique(uvd.baseline_array))
for i, red_group in enumerate(reds):
    true_vis_red = true_vis[i]
    for bl in red_group:
        # Conjugate the visibilities if the baseline is conjugately redundant
        if bl in conj_bls:
            true_vis_red = true_vis_red.conj()

        # Add noise and RFI
        noise_sim = noise.sky_noise_jy(Tsky + Trx, freqs, lsts)
        noisy_vis = true_vis_red + noise_sim + rfi_all

        # Add crosstalk
        xtalk = sigchain.gen_xtalk(freqs)
        xtalk_noisy_vis = noisy_vis + xtalk

        # Apply antenna gains and save
        bl_tuple = uvd.baseline_to_antnums(bl)
        g_ij = gains[bl_tuple[0]] * gains[bl_tuple[1]].conj()
        if bl in bl_dict.keys():
            bl_dict[bl] = xtalk_noisy_vis * g_ij
        else:
            print('{0} is not in the baseline dictionary.'.format(bl))
# Set any missed baselines to NaNs (should just be the autos)
# NOTE: Again, there's probably a better way of dealing with this.
print('Setting the following baselines to NaNs:')
for bl in bl_dict.keys():
    if bl_dict[bl] is None:
        bl_dict[bl] = np.full((len(lsts), len(freqs)), np.nan + 1j*np.nan)
        print('\t{0}'.format(uvd.baseline_to_antnums(bl)))

# Fit the model into the UVData structure
sim_data = np.zeros_like(uvd.data_array)
for bl in bl_dict.keys():
    blt_ind = np.where(uvd.baseline_array == bl)[0]
    sim_data[blt_ind] = bl_dict[bl][:, None, :, None]

# Save to a miriad file
# Unflag everything just in case anything was flagged
uvd_sim.flag_array = np.full_like(uvd.flag_array, False)
uvd_sim.data_array = sim_data
uvd_sim.write_miriad(sim_out)
