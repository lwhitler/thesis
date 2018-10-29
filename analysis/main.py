import numpy as np
import matplotlib.pyplot as plt
import copy
import hera_pspec as hp
from pyuvdata import UVData
from hera_qm.metrics_io import process_ex_ants
import utils

# Paths
JD_dec = '49088'
time_thresh_str = '0.1'
base_name = 'zen.2458106.' + JD_dec + '.xx.HH'
beam_file = '/home/lwhitler/data/dfiles/HERA_NF_dipole_power.beamfits'
data_file = '/data6/HERA/data/IDR2.1/2458106/' + base_name + '.uvOCRS'
ant_metrics = '/data6/HERA/data/IDR2.1/2458106/' + base_name.replace('.xx','') + '.uv.ant_metrics.json'
uvp_prefix = '/home/lwhitler/data/dfiles/pspecs/broadcasting/tt' + time_thresh_str + '/' + base_name + ''

# Cosmology and beam models
cosmo = hp.conversions.Cosmo_Conversions()
psbeam = hp.pspecbeam.PSpecBeamUV(beam_file, cosmo=cosmo)
# Load the data
uvd_orig = UVData()
uvd_orig.read_miriad(data_file)
uvd_default, uvd = copy.deepcopy(uvd_orig), copy.deepcopy(uvd_orig)
# Convert data from Jy to mK
utils.aux.convert_Jy_to_mK(uvd_default, psbeam)
utils.aux.convert_Jy_to_mK(uvd, psbeam)

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
# Time threshold for broadcasting flags
time_thresh = float(time_thresh_str)

# Make/load the power spectra
uvp_default = uvp_prefix.replace(time_thresh_str, '0.2') + '.ps.' + spw_str + '.' + bl_str + '.' + uvd.vis_units + '.h5'
uvp_file = uvp_prefix + '.ps.' + spw_str + '.' + bl_str + '.' + uvd.vis_units + '.h5'
ds_default, uvp_default = utils.aux.get_uvpspec(uvd_default, psbeam,
                                                bls1, bls2, spw=0,
                                                uvp_default)
ds, uvp = utils.aux.get_uvpspec(uvd, psbeam, uvp_file)

# Take the time average of the spectra
uvp_default_avg = uvp_default.average_spectra(time_avg=True, inplace=False)
uvp_avg = uvp.average_spectra(time_avg=True, inplace=False)
# Save the baseline pairs
blpairs = list(np.unique(uvp_avg.blpair_array))

# Find all nonzero data and corresponding baselines
zero_wgt = np.all(uvp_avg.wgt_array[0][:, :, 0] == 0, axis=1)
zero_wgt_mask = np.broadcast_to(zero_wgt, uvp_avg.data_array[0][:, :, 0].shape)
zero_wgt_mask = zero_wgt_mask[:, :, np.newaxis]
nonzero_blpairs = uvp_avg.blpair_array[~zero_wgt[:, 0]]
uvp_default_avg.data_array[0] = np.ma.masked_array(uvp_default_avg.data_array[0], zero_wgt_mask)
uvp_avg.data_array[0] = np.ma.masked_array(uvp_avg.data_array[0], zero_wgt_mask)

# The four panel plot with flags before and after broadcasting, spectra of
# all baseline pairs, and the median power spectrum with errors
fig, ax = plt.subplots(2, 2, figsize=(12, 12))
utils.plot.plot_flag_frac(uvd_orig, good_bls, ax[0, 0], vmin=0, vmax=1)
utils.plot.plot_flag_frac(ds.dsets[0], good_bls, ax[0, 1], vmin=0, vmax=1)
utils.plot.plot_multiple_blpairs(uvp_avg, ax[1, 0], blpairs=nonzero_blpairs)
utils.plot.plot_median_spectra(uvp_default_avg, ax[1, 1],
                               blpairs=nonzero_blpairs, color='#0700FF',
                               label='Default (time threshold: 0.2)')
utils.plot.plot_median_spectra(uvp_avg, ax[1, 1], blpairs=nonzero_blpairs,
                               color='#8600FF', label='Time threshold: {}'.format(time_thresh_str))
# Plot appearance
ax[1, 1].legend(loc='best')
fig.canvas.draw()
zero_index = utils.plot.find_zero_tick_label(ax[1, 0])
if zero_index is not None:
    ax[1, 0].yaxis.get_major_ticks()[zero_index].label.set_visible(False)
zero_index = utils.plot.find_zero_tick_label(ax[1, 1])
if zero_index is not None:
    ax[1, 1].yaxis.get_major_ticks()[zero_index].label.set_visible(False)
ax[0, 0].set_title('Original flags')
ax[0, 1].set_title('Flags after broadcasting')
plt.show()
