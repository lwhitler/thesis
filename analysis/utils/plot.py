import numpy as np
import matplotlib.pyplot as plt
import hera_pspec as hp
import aux


def plot_multiple_blpairs(uvp, ax, blpairs=None, plot_median=True, delay=False,
                          yscale='symlog', hline=True, **kwargs):
    """
    Plot the power spectra from multiple baseline pairs.

    Parameters
    ----------
    uvp : UVPSpec object
        UVPSpec object containing the power spectra to plot
    ax : Axes object
        Axes object to plot the power spectra on
    blpairs : list(s) of tuples, optional
        The baseline pairs to plot the spectra of (default is all of the
        baseline pairs in the UVPSpec object)
    plot_median : bool, optional
        Whether to plot the median of the baseline pair power spectra (default
        is True)
    delay : bool, optional
        Whether to plot in delay (ns) or cosmological units (h Mpc^-1) (default
        is cosmological units)
    yscale : str, optional
        The y-axis scale ('linear', 'log', or 'symlog', default is symlog)
    hline : bool, optional
        Whether to plot a horizontal line at zero (default is True)
    """
    # Default to baseline pairs in UVPSpec object
    if blpairs is None:
        blpairs = list(np.unique(uvp.blpair_array))

    if hline:
        ax.axhline(0, c='#444444', ls=':', lw=0.75)
        hline = False
    hp.plot.delay_spectrum(uvp, blpairs, spw=0, pol='xx', delay=delay,
                           force_plot=True, logscale=False, c='#383838',
                           lw=0.25, alpha=0.01, ax=ax)
    if plot_median:
        plot_median_spectra(uvp, ax, delay=delay, hline=hline)

    # y-axis scaling
    if yscale == 'symlog':
        linthreshy = np.max(np.real(uvp.data_array[0]))*1e-4
        ax.set_yscale(yscale, linthreshy=linthreshy)
    else:
        ax.set_yscale(yscale)

    # Axis labeling
    xlabel, ylabel = make_axis_labels(uvp, delay=delay)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)


def plot_flag_frac(uvd, bls, ax, **kwargs):
    """
    Plot the waterfall for the percentage of flagged baselines.

    Parameters
    ----------
    uvd : UVData object
        The UVData object containing the flags
    bls : array-like
        Baselines for which to look for flags
    ax : Axes object
        Axes object to plot the waterfall on
    """
    flag_frac = aux.calc_flagged_bl_percent(uvd, bls)
    ax.imshow(flag_frac, aspect='auto', **kwargs)


def plot_median_spectra(med, med_err, uvp, ax, delay=False, yscale='symlog',
                        hline=True, **kwargs):
    """
    Plot the median power spectra of multiple baseline pairs.

    Parameters
    ----------
    med : array-like
        The median to plot
    med_err : array-like
        The error on the median
    uvp : UVPSpec object
    ax : Axes object
        Axes object to plot the median in
    blpairs : list(s) of tuples, optional
        The baseline pairs to plot the spectra of (default is all of the
        baseline pairs in the UVPSpec object)
    niters : int, optional
        Number of resamples to take to calculate the error of the median
        (default is 1000)
    delay : bool, optional
        Whether to plot in delay (ns) or cosmological units (h Mpc^-1) (default
        is cosmological units)
    yscale : str, optional
        The y-axis scale ('linear', 'log', or 'symlog', default is symlog)
    hline : bool, optional
        Whether to plot a horizontal line at zero (default is True)
    """
    # Get the x-axis units
    if not delay:
        x = uvp.get_kparas(0)  # k_parallel in h Mpc^-1
    else:
        x = uvp.get_dlys(0) * 1e9  # delay in ns

    # Plot median and errors
    if hline:
        ax.axhline(0, c='#444444', ls=':', lw=0.75)
    ax.fill_between(x, med+med_err, med-med_err, alpha=0.3, **kwargs)
    ax.plot(x, median, lw=1.25, **kwargs)

    # y-axis scaling
    if yscale == 'symlog':
        linthreshy = np.max(np.real(uvp.data_array[0]))*1e-5
        linthreshy = 10**np.floor(np.log10(linthreshy))
        ax.set_yscale(yscale, linthreshy=linthreshy,
                      linscaley=2)
    else:
        ax.set_yscale(yscale)

    # Axis labeling
    xlabel, ylabel = make_axis_labels(uvp, delay=delay)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)


def make_axis_labels(uvp, delay=False):
    """
    Make the axis labels for plotting the power spectra.

    Parameters
    ----------
    uvp : UVPSpec object
        UVPSpec object containing the power spectra
    delay : bool, optional
        Whether the spectra are in delay (ns) or cosmological units (h Mpc^-1)
        (default is cosmological units)
    """
    if delay:
        xlabel = r'$\tau$ $[{\rm ns}]$'
    else:
        xlabel = r'k$_\parallel$ h Mpc$^{-1}$'
    vis_units = uvp.vis_units
    ylabel = r'P(k$_\parallel$) [(' + vis_units + '$^2$ h$^{-3}$ Mpc$^3$]'
    return xlabel, ylabel


def find_zero_tick_label(ax):
    """
    Find the tick label at y = 0.

    Parameters
    ----------
    ax : Axes object
        Axes object containing the ticks
    """
    yticks = ax.yaxis.get_major_ticks()
    for i in range(len(yticks)):
        text = yticks[i].label.get_text()
        if text == '$\mathdefault{0}$':
            return i
