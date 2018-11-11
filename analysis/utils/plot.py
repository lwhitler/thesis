import numpy as np
import matplotlib.pyplot as plt
import hera_pspec as hp
import aux


def plot_multiple_blpairs(uvp, ax, blpairs=None, delay=False,
                          vis_units='mK', hline=True, **kwargs):
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
    delay : bool, optional
        Whether to plot in delay (ns) or cosmological units (h Mpc^-1) (default
        is cosmological units)
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

    # Axis labeling
    xlabel, ylabel = make_axis_labels(delay=delay, vis_units=vis_units)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)


def plot_flag_frac(uvd, bls, ax, spw=(0, 1024), **kwargs):
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
    spw : tuple, optional
        The spectral window to plot (default is the entire band)
    """
    flag_frac = aux.calc_flagged_bl_percent(uvd, bls)
    ax.imshow(flag_frac[:, spw[0]:spw[1]], aspect='auto', **kwargs)

    # Axis labeling
    # Channels corresponding to every 2 MHz between 100 and 200 MHz
    full_band_chans = np.array([0., 20.48, 40.96, 61.44, 81.92, 102.4,
                                122.88, 143.36, 163.84, 184.32, 204.8,
                                225.28, 245.76, 266.24, 286.72, 307.2,
                                327.68, 348.16, 368.64, 389.12, 409.6,
                                430.08, 450.56, 471.04, 491.52, 512.,
                                532.48, 552.96, 573.44, 593.92, 614.4,
                                634.88, 655.36, 675.84, 696.32, 716.8,
                                737.28, 757.76, 778.24, 798.72, 819.2,
                                839.68, 860.16, 880.64, 901.12, 921.6,
                                942.08, 962.56, 983.04, 1003.52, 1024.])
    spw_chans = full_band_chans[(full_band_chans >= spw[0]) & (full_band_chans <= spw[1])]
    spw_freqs = [str(int(aux.chan_to_freqs(chan))) for chan in spw_chans]
    ax.set_xticks(spw_chans - spw[0])
    ax.set_xticklabels(spw_freqs)
    ax.set_xlabel('Frequency [MHz]', fontsize=12)
    ax.set_ylabel('Time', fontsize=12)


def plot_median_spectra(x, med, med_err, ax, delay=False, vis_units='mK',
                        hline=True, color=None, label=None, **kwargs):
    """
    Plot the median power spectra of multiple baseline pairs.

    Parameters
    ----------
    x : array-like
        The x-values to plot (either delay or cosmological units)
    med : array-like
        The median to plot
    med_err : array-like
        The error on the median
    ax : Axes object
        Axes object to plot the median in
    delay : bool, optional
        Whether the spectrum is plotted in delay (ns) or cosmological units
        (h Mpc^-1) (default is cosmological units)
    vis_units : str, optional
        Units of visibility data for axis labelling (default is 'mK')
    hline : bool, optional
        Whether to plot a horizontal line at zero (default is True)
    color : str
        What color to plot the spectrum in
    label : str
        Label for the legend
    """
    # Plot median and errors
    if hline:
        ax.axhline(0, c='#444444', ls=':', lw=0.75)
    ax.fill_between(x, med+med_err, med-med_err, alpha=0.3, color=color)
    ax.plot(x, med, lw=1.25, color=color, label=label, **kwargs)

    # Axis labeling
    xlabel, ylabel = make_axis_labels(delay=delay, vis_units=vis_units)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)


def make_axis_labels(delay=False, vis_units='mK'):
    """
    Make the axis labels for plotting the power spectra.

    Parameters
    ----------
    delay : bool, optional
        Whether the spectra are in delay (ns) or cosmological units (h Mpc^-1)
        (default is cosmological units)
    vis_units : str, optional
        Units of visibility data for axis labelling (default is 'mK')
    """
    if delay:
        xlabel = r'$\tau$ $[{\rm ns}]$'
    else:
        xlabel = r'k$_\parallel$ h Mpc$^{-1}$'
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
