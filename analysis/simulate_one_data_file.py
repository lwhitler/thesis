import numpy as np
import matplotlib.pyplot as plt
import colormaps as cmaps
import cmocean
import copy
from scipy import constants
from pyuvdata import UVData
from pyuvdata import utils as uvutils
from hera_sim import foregrounds, noise, rfi, sigchain


# Some useful functions
def make_amp_phase_plot(data, fig, ax, log=True, **kwargs):
    if log:
        data = np.log10(np.abs(data))
        title = 'log(Amplitude)'
    else:
        data = np.abs(data)
        title = 'Amplitude'
    cax0 = ax[0].imshow(data, cmap=cmaps.inferno, aspect='auto', **kwargs)
    cax1 = ax[1].imshow(np.angle(data), cmap='cmo.phase', vmin=-np.pi,
                        vmax=np.pi, aspect='auto')
    ax[0].set_title(title)
    ax[1].set_title('Phase')
    fig.colorbar(cax0, ax=ax[0])
    fig.colorbar(cax1, ax=ax[1])


def bl_len_in_ns(uvw_arr):
    return (np.linalg.norm(uvw_arr) / 299792458.) * 1e9


# Paths
JD_dec = '40141'
base_name = 'zen.2458106.' + JD_dec + '.xx.HH.uv'
data_in = '/data6/HERA/data/IDR2.1/2458106/' + base_name
sim_out = '/home/lwhitler/data/dfiles/sim/' + base_name.replace('.uv', '') + '.sim.uv'
