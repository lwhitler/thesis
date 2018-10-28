import numpy as np
import matplotlib.pyplot as plt
import copy
import os.path
import hera_pspec as hp
from pyuvdata import UVData
from hera_qm.metrics_io import process_ex_ants
import utils

# Paths
JD_dec = '49088'
beam_file = '/home/lwhitler/data/dfiles/HERA_NF_dipole_power.beamfits'  # noqa
data_file = '/data6/HERA/data/IDR2.1/2458106/zen.2458106.' + JD_dec + '.xx.HH.uvOCRS'  # noqa
ant_metrics = '/data6/HERA/data/IDR2.1/2458106/zen.2458106.' + JD_dec + '.HH.uv.ant_metrics.json'  # noqa
uvp_prefix = '/home/lwhitler/data/dfiles/pspecs/baselines/zen.2458106.' + JD_dec + '.xx.HH.uvOCRS'  # noqa

# Cosmology and beam models
cosmo = hp.conversions.Cosmo_Conversions()
uvb = hp.pspecbeam.PSpecBeamUV(beam_file, cosmo=cosmo)
# Load the data
uvd_orig = UVData()
uvd_orig.read_miriad(data_file)
uvd = copy.deepcopy(uvd_orig)
# Convert data from Jy to mK
freq_array = np.unique(uvd.freq_array)
uvd.data_array *= uvb.Jy_to_mK(freq_array, pol='xx')[None, None, :, None]
uvd.vis_units = 'mK'

# Select the baseline pairs without bad antennas
reds = hp.utils.get_reds(uvd)[0]
bls = reds[2]  # 14-m EW baselines
bl_str = 'EW_14m'  # For saving files
xants = process_ex_ants(metrics_file=ant_metrics)  # Bad antennas
good_bls = utils.aux.find_good_bls(bls, xants)
blps = hp.utils.construct_blpairs(good_bls, exclude_auto_bls=True,
                                  exclude_permutations=True)
bls1, bls2 = blps[0], blps[1]

# Select the spectral window
spw = [(512, 640)]
spw_str = str(spw[0][0]) + '-' + str(spw[0][1])  # For saving files

# Make/load the power spectrum
uvp_file = uvp_prefix + '.ps.' + spw_str + '.' + bl_str + '.' + uvd.vis_units + '.h5'  # noqa
ds = hp.pspecdata.PSpecData([uvd, uvd], wgts=[None, None], beam=uvb)
ds.broadcast_dset_flags()
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
    uvp.write_hdf5(uvp_file)

# Take the time average of the spectra
uvp_avg = uvp.average_spectra(time_avg=True, inplace=False)
# Save the baseline pairs
blpairs = list(np.unique(uvp_avg.blpair_array))

# Find all nonzero data and corresponding baselines
zero_wgt = np.all(uvp_avg.wgt_array[0][:, :, 0] == 0, axis=1)
zero_wgt_mask = np.broadcast_to(zero_wgt, uvp_avg.data_array[0][:, :, 0].shape)
zero_wgt_mask = zero_wgt_mask[:, :, np.newaxis]
nonzero_blpairs = uvp_avg.blpair_array[~zero_wgt[:, 0]]
uvp_avg.data_array[0] = np.ma.masked_array(uvp_avg.data_array[0], zero_wgt_mask)

# The four panel plot with flags before and after broadcasting, spectra of
# all baseline pairs, and the median power spectrum with errors
fig, ax = plt.subplots(2, 2, figsize=(12, 12))
utils.plot.plot_flag_frac(uvd_orig, good_bls, ax[0, 0], vmin=0, vmax=1)
utils.plot.plot_flag_frac(ds.dsets[0], good_bls, ax[0, 1], vmin=0, vmax=1)
utils.plot.plot_multiple_blpairs(uvp_avg, ax[1, 0], blpairs=nonzero_blpairs)
utils.plot.plot_median_spectra(uvp_avg, ax[1, 1], blpairs=nonzero_blpairs)
# Axis labeling and titles
ax[0, 0].set_title('Original flags')
ax[0, 1].set_title('After broadcasting')
ax[1, 1].yaxis.label.set_visible(False)
plt.show()
