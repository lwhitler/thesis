import numpy as np
import matplotlib.pyplot as plt
import hera_pspec as hp
import utils


def plot_all_blps(uvp, ax, spw, blps=None, plot_median=True, delay=False,
                  yscale='symlog', hline=False):
    """
    """
    kparas = uvp.get_kparas(0)
    # Default to baseline pairs in UVPSpec object
    if blps is None:
        blps = list(np.unique(uvp.blpair_array))

    if hline:
        ax.axhline(0, c='#444444', ls=':')
        hline = False
    hp.plot.delay_spectrum(uvp, blps, spw=spw, pol='XX', delay=delay,
                           force_plot=True, logscale=False, c='#383838',
                           lw=0.25, alpha=0.01, ax=ax)
    if plot_median:
        plot_median_spectra(uvp, ax, spw, delay=delay, hline=hline)

    # y-axis scaling
    if yscale == 'symlog':
        linthreshy = np.max(np.real(uvp.data_array[0]))*1e-4
        ax.set_yscale(yscale, linthreshy=linthreshy)
    else:
        ax.set_yscale(yscale)


def plot_flag_frac(uvd, bls, ax, **kwargs):
    """
    Plot the waterfall for the percentage of flagged baselines.
    """
    flag_frac = utils.calc_flagged_bl_percent(uvd, bls)
    ax.imshow(flag_frac, **kwargs)


def plot_median_spectra(uvp, spw, ax, blpairs=None, niters=1000, delay=False,
                        yscale='symlog', hline=False):
    """
    """
    # Get the x axis units
    if not delay:
        x = uvp.get_kparas(spw)  # k_parallel in h^-1 Mpc
    else:
        x = uvp.get_dlys(spw) * 1e9  # delay in ns

    # Get the median and bootstrap for errors
    median = utils.calc_median(uvp)
    med_sd = utils.bootstrap_median(uvp, blpairs=blpairs, niters=niters)

    # Plot median and errors
    if hline:
        ax.axhline(0, c='#444444', ls=':', lw=0.75)
    ax.fill_between(x, median+med_sd, median-med_sd, color='#0700FF',
                    alpha=0.25)
    ax.plot(x, median, c='#0700FF', lw=1.25)

    # y-axis scaling
    if yscale == 'symlog':
        linthreshy = np.max(np.real(uvp.data_array[0]))*1e-3
        ax.set_yscale(yscale, linthreshy=linthreshy)
    else:
        ax.set_yscale(yscale)
