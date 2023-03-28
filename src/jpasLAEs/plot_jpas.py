import numpy as np
import matplotlib.pyplot as plt

from jpasLAEs.utils import w_central, data_tab

def plot_JPAS_source(flx, err, ax=None,
                     set_ylim=True, e17scale=False, fs=15):
    '''
    Generates a plot with the JPAS data.

    Parameters:
    -----------
    flx : array_like
        Array of flux values to plot.
    err : array_like
        Array of error values to plot.
    set_ylim : bool, optional
        Whether to set the y-axis limits to match the data range. Default is True.
    e17scale : bool, optional
        Whether to scale the y-axis by 1e17. Default is False.
    fs : int, optional
        Font size for axis labels. Default is 15.

    Returns:
    --------
    ax : matplotlib axis
        Axis object containing the plot.
    '''

    if e17scale:
        flx = flx * 1e17
        err = err * 1e17

    cmap = data_tab['color_representation']

    data_max = np.max(flx)
    data_min = np.min(flx)
    y_max = (data_max - data_min) * 2/3 + data_max
    y_min = data_min - (data_max - data_min) * 0.3

    # if not given any ax, create one
    if ax is None:
        ax = plt.gca()

    for i, w in enumerate(w_central[:-4]):
        ax.errorbar(w, flx[i], yerr=err[i],
                    marker='o', markeredgecolor='dimgray', markerfacecolor=cmap[i],
                    markersize=8, ecolor='dimgray', capsize=4, capthick=1, linestyle='',
                    zorder=-99)
    ax.errorbar(w_central[-4], flx[-4], yerr=err[-4], markeredgecolor='dimgray',
                fmt='s', markerfacecolor=cmap[-4], markersize=10,
                ecolor='dimgray', capsize=4, capthick=1)
    ax.errorbar(w_central[-3], flx[-3], yerr=err[-3], markeredgecolor='dimgray',
                fmt='s', markerfacecolor=cmap[-3], markersize=10,
                ecolor='dimgray', capsize=4, capthick=1)
    ax.errorbar(w_central[-2], flx[-2], yerr=err[-2], markeredgecolor='dimgray',
                fmt='s', markerfacecolor=cmap[-2], markersize=10,
                ecolor='dimgray', capsize=4, capthick=1)
    ax.errorbar(w_central[-1], flx[-1], yerr=err[-1], markeredgecolor='dimgray',
                fmt='s', markerfacecolor=cmap[-1], markersize=10,
                ecolor='dimgray', capsize=4, capthick=1)

    try:
        if set_ylim:
            ax.set_ylim((y_min, y_max))
    except:
        pass

    ax.set_xlabel('$\lambda\ (\AA)$', size=fs)
    if e17scale:
        ax.set_ylabel(
            r'$f_\lambda\cdot10^{17}$ (erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$)', size=fs)
    else:
        ax.set_ylabel(
            '$f_\lambda$ (erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$)', size=fs)

    return ax