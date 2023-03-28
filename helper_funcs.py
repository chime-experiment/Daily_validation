import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os
from draco.core import containers
from draco.util import tools
from ch_util import ephemeris
from chimedb import core, dataflag as df
import copy

#==========================================================================

def plotDS(rev, LSD, hpf=False, vmin=-3, vmax=2, cmap='inferno'):
    """
    Plots the delay spectrum for a given LSD.
    
    Parameters
    ----------
    rev : int
          Revision number
    LSD : int
          Day number
    hpf : bool, optional (default False)
          with/without high pass filter (True/False)
    vmin, vmax : min, max values in the colorscale, optional 
    cmap : colormap, optional (default 'inferno')
    
    Returns
    -------
    Delay Spectrum
    """
    
    path = "/project/rpp-chime/chime/chime_processed/daily/"
    path += "rev_0" + str(rev) + "/" + str(LSD)
    
    if hpf == True:
        data = "delayspectrum_hpf_lsd_" + str(LSD) + ".h5"    
    else:
        data = "delayspectrum_lsd_" + str(LSD) + ".h5"
        
    DelaySpec = containers.DelaySpectrum.from_file(os.path.join(path, data))
    
    DS = np.array(DelaySpec.spectrum)
    
    plt.figure(figsize=(15,5))
    ax = plt.gca()
    im = ax.imshow(np.log10(DS.T), vmin = vmin, vmax = vmax, 
                   cmap=cmap, origin='lower', aspect = 'auto');
    divider = make_axes_locatable(ax);
    cax = divider.append_axes("right", size="1.5%", pad=0.25);

    cb = plt.colorbar(im, cax=cax);
    
    xind = [0, 500, 1000, 1500];
    ax.set_xticks(xind);
    ax.set_xlabel('Baseline')
    ax.set_ylabel('Delay')
    title = 'rev 0' + str(rev) + ', LSD ' + str(LSD) + ', hpf = ' + str(hpf)
    ax.set_title(title, fontsize=20)
    
    del DelaySpec, DS
    pass


#========================================================================

def plotRingmap(rev, LSD, vmin=-5, vmax=20, cmap='inferno', flag_mask = True):
    """
    Plots the delay spectrum for a given LSD.
    
    Parameters
    ----------
    rev : int
          Revision number
    LSD : int
          Day number
    vmin, vmax : min, max values in the colorscale, optional 
    cmap : colormap, optional (default 'inferno')
    
    Returns
    -------
    Ringmap
    """
    
    path = "/project/rpp-chime/chime/chime_processed/daily/"
    path += "rev_0" + str(rev) + "/" + str(LSD) + "/" 
    data = "ringmap_lsd_" + str(LSD) + ".zarr.zip" 
    ringmap = containers.RingMap.from_file(
            os.path.join(path, data), freq_sel=slice(399,401))
       
    freq = ringmap.freq
    indx_midfrq = freq.shape[0] // 2
    
    m = ringmap.map[0, 0, indx_midfrq].T[::-1]
    w = ringmap.weight[0, indx_midfrq].T[::-1]

    nanmask = np.where(w == 0, np.nan, 1)
    nanmask *= np.where(_mask_flags(ephemeris.csd_to_unix(LSD + ringmap.ra / 360.0), LSD), np.nan, 1)

    m -= np.nanmedian(m * nanmask, axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(15,10))
    ax = plt.gca()
    
    if flag_mask == True:
        im = ax.imshow(m * nanmask, vmin=vmin, vmax=vmax, 
                       aspect="auto", origin="lower", extent=(0, 360, -1, 1), cmap=cmap)
    else:
        im = ax.imshow(m, vmin=vmin, vmax=vmax, 
                       aspect="auto", origin="lower", extent=(0, 360, -1, 1), cmap=cmap)
    ax.set_xlabel("RA [degrees]")
    ax.set_ylabel("sin(ZA)")
    divider = make_axes_locatable(ax);
    cax = divider.append_axes("right", size="1.5%", pad=0.25);
    cb = plt.colorbar(im, cax=cax);
    title = ('rev 0' + str(rev) + ', LSD ' + str(LSD) + 
            f', {freq[freq.shape[0]//2]:.2f}' + ' MHz')
    ax.set_title(title, fontsize=20)
    
    del ringmap, freq
    pass

#========================================================================

def flag_time_spans(LSD):
    core.connect()

    ut_start = ephemeris.csd_to_unix(LSD)
    ut_end = ephemeris.csd_to_unix(LSD + 1)

    bad_flags = [
        'bad_calibration_fpga_restart',
        #'globalflag',
        #'acjump',
        'acjump_sd',
        #'rain',
        #'rain_sd',
        'bad_calibration_acquisition_restart',
        #'misc',
        #'rain1mm',
        'rain1mm_sd',
        'srs/bad_ringmap_broadband',
        'bad_calibration_gains',
        'snow',
        'decorrelated_cylinder',
    ]

    flags = (
        df.DataFlag.select().where(
            df.DataFlag.start_time < ut_end,
            df.DataFlag.finish_time > ut_start
        ).join(df.DataFlagType)
        .where(df.DataFlagType.name << bad_flags)
    )

    flag_time_spans = [(f.type.name, f.start_time, f.finish_time) for f in flags]
    
    return flag_time_spans


def _mask_flags(times, LSD):
    flag_mask = np.zeros_like(times, dtype=np.bool)

    for type_, ca, cb in flag_time_spans(LSD):
        flag_mask[(times > ca) & (times < cb)] = True
    
    return flag_mask


def plotSens(rev, LSD, vmin = 0.995, vmax = 1.005):
    path = "/project/rpp-chime/chime/chime_processed/daily/"
    path += "rev_0" + str(rev) + "/" + str(LSD)

    data = "sensitivity_lsd_" + str(LSD) + ".h5"

    sens = containers.SystemSensitivity.from_file(
        os.path.join(path, data))
    
    data = "rfi_mask_lsd_" + str(LSD) + ".h5"

    rfm = containers.RFIMask.from_file(
        os.path.join(path, data))

    sp = 0
    fig, axis = plt.subplots(1, 1, figsize=(15, 10))

    sensrat = sens.measured[:, sp] * tools.invert_no_zero(sens.radiometer[:, sp])
    sensrat /= np.median(sensrat, axis=1)[:, np.newaxis]
    sensrat *= np.where(rfm.mask[:] == 0, 1, np.nan)
    sensrat *= np.where(_mask_flags(sens.time, LSD), np.nan, 1)

    cmap = copy.copy(matplotlib.cm.viridis)
    cmap.set_bad("#aaaaaa")

    im = axis.imshow(sensrat, extent=(0, 360, 400, 800), cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(axis);
    cax = divider.append_axes("right", size="1.5%", pad=0.25);
    cb = plt.colorbar(im, cax=cax);
    #fig.colorbar(im)
    axis.set_xlabel("RA [deg]")
    axis.set_ylabel("Freq [MHz]")
    title = ('rev 0' + str(rev) + ', LSD ' + str(LSD))
    axis.set_title(title, fontsize=20)
    _ = axis.set_xticks(np.arange(0, 361, 45))

#========================================================================

def circle(ax, x, y, radius, **kwargs):
    """Create circle on figure with axes of different sizes.

    Plots a circle on the current axes using `plt.Circle`, taking into account
    the figure size and the axes units.

    It is done by plotting in the figure coordinate system, taking the aspect
    ratio into account. In this way, the data dimensions do not matter.
    However, if you adjust `xlim` or `ylim` after plotting `circle`, it will
    screw them up; set `plt.axis` before calling `circle`.

    Parameters
    ----------
    ax : axis
        Matplotlib axis to plot the circle against.
    xy, radius, kwars :
        As required for `plt.Circle`.
        
    Returns
    -------
    circle : Artist
    """
    from matplotlib import patches
    
    fig = ax.figure
    trans = fig.dpi_scale_trans + matplotlib.transforms.ScaledTranslation(x, y, ax.transData)
    circle = patches.Circle((0, 0), radius, transform=trans, **kwargs)
    
    # Draw circle
    return ax.add_artist(circle)

#========================================================================

def imshow_sections(axis, x, y, c, gap_scale=0.1, *args, **kwargs):
    """Plot an array with the pixels at given locations accounting for gaps.
    
    Parameters
    ----------
    axis : matplotlib.Axis
        Axis to show image in.
    x, y : np.ndarray[:]
        Location of pixel centres in each direction
    c : np.ndarray[:, :]
        Pixel values
    gap_scale : float, optional
        If there is an extra gap between pixels of this amount times the nominal
        separation, consider this a gap in the data.
        
    Returns
    -------
    artists : list
        List of the artists for each image section.
    """
    
    def _find_splits(ax):
        d = np.diff(ax)
        md = np.median(d)
        
        ranges = []
        
        last_cut = 0
        for ii, di in enumerate(d):
            if np.abs(di - md) > np.abs(gap_scale * md):
                ranges.append((last_cut, ii+1))
                last_cut = ii + 1
        
        ranges.append((last_cut, len(ax)))
        
        return ranges
        
    artists = []
    for xs, xe in _find_splits(x):
        for ys, ye in _find_splits(y):
            
            xa = x[xs:xe]
            ya = y[ys:ye]
            ca = c[ys:ye, xs:xe]
            
            artists.append(axis.imshow(ca, extent=(xa[0], xa[-1], ya[-1], ya[0]), *args, **kwargs))
            
    return artists
            
