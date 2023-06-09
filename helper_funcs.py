import os
import copy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skyfield import almanac

from caput import weighted_median
from caput import time as ctime
from draco.core import containers
from draco.util import tools
from draco.analysis.sidereal import _search_nearest
from ch_util import ephemeris, rfi
from chimedb import core, dataflag as df

eph = ephemeris.skyfield_wrapper.ephemeris
chime_obs = ephemeris.chime
sf_obs = chime_obs.skyfield_obs()

#==========================================================================

def plotDS(rev, LSD, hpf=False, clim = [1e-3, 1e2], cmap='inferno'):
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
        
    DS = containers.DelaySpectrum.from_file(os.path.join(path, data))
    
    tau = DS.index_map["delay"] * 1e3
    DS_Spec = DS.spectrum
    
    baseline_vec = DS.index_map["baseline"] 
    bl_mask = np.zeros((4, baseline_vec.shape[0]), dtype=bool)
    bl_mask[0] = baseline_vec[:, 0] < 10
    bl_mask[1] = (baseline_vec[:, 0] > 10) & (baseline_vec[:, 0] < 30)
    bl_mask[2] = (baseline_vec[:, 0] > 30) & (baseline_vec[:, 0] < 50)
    bl_mask[3] = (baseline_vec[:, 0] > 50)
    
    
    fig, ax = plt.subplots(1, 4, figsize=(15,8), sharey=True,
                          gridspec_kw={'width_ratios': [1, 2, 2, 2]})
    imshow_params = {"origin": "lower", "aspect": "auto", "interpolation": "none", "norm": LogNorm(), "clim": clim, "cmap": cmap}
    
    for i in range(4):
        baseline_idx_sorted = baseline_vec[bl_mask[i], 1].argsort()
        extent = [baseline_vec[bl_mask[i], 1].min(), 
                  baseline_vec[bl_mask[i], 1].max(), tau[0], tau[-1]]
        im = ax[i].imshow(DS_Spec[bl_mask[i]][baseline_idx_sorted].T.real,
                          extent=extent, **imshow_params)
        ax[i].xaxis.set_tick_params(labelsize=18)
        ax[i].yaxis.set_tick_params(labelsize=18)
        ax[i].set_title(f"{i}-cyl", fontsize=20)
        
    #ax[0].set_ylabel("Delay [ns]")
    fig.supxlabel("NS baseline length [m]", fontsize = 20)
    fig.supylabel("Delay [ns]", fontsize = 20)
    title = 'rev 0' + str(rev) + ', LSD ' + str(LSD) + ', hpf = ' + str(hpf)
    fig.suptitle(title,  fontsize = 20)
    fig.subplots_adjust(wspace=0.05)
    fig.colorbar(im, ax=ax, orientation='vertical', pad = 0.02, aspect=40)

    
    del DS
    pass


#========================================================================

def plotRingmap(rev, LSD, vmin=-5, vmax=20, fi=400, flag_mask = True):
    """
    Plots the delay spectrum for a given LSD.
    
    Parameters
    ----------
    rev : int
          Revision number
    LSD : int
          Day number
    vmin, vmax : min, max values in the colorscale, optional
    fi  : freq index, optional
    cmap : colormap, optional (default 'inferno')
    
    Returns
    -------
    Ringmap
    """
    
    path = "/project/rpp-chime/chime/chime_processed/daily/"
    path += "rev_0" + str(rev) + "/" + str(LSD) + "/" 
    data = "ringmap_lsd_" + str(LSD) + ".zarr.zip"
    
    ringmap = containers.RingMap.from_file(
            os.path.join(path, data), freq_sel=slice(fi, fi+1))
    
    freq = ringmap.freq
    m = ringmap.map[0, 0, 0].T[::-1]
    w = ringmap.weight[0, 0].T[::-1]

    nanmask = np.where(w == 0, np.nan, 1)
    nanmask *= np.where(_mask_flags(ephemeris.csd_to_unix(LSD + ringmap.ra / 360.0), LSD), np.nan, 1)

    m -= np.nanmedian(m * nanmask, axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(15,10))
    ax = plt.gca()
    
    cmap = copy.copy(matplotlib.cm.inferno)
    cmap.set_bad("grey")
    
    if flag_mask == True:
        im = ax.imshow(m * nanmask, vmin=vmin, vmax=vmax, 
                       aspect="auto", extent=(0, 360, -1, 1), cmap=cmap)
    else:
        im = ax.imshow(m, vmin=vmin, vmax=vmax, 
                       aspect="auto", extent=(0, 360, -1, 1), cmap=cmap)
    ax.set_xlabel("RA [degrees]")
    ax.set_ylabel("sin(ZA)")
    divider = make_axes_locatable(ax);
    cax = divider.append_axes("right", size="1.5%", pad=0.25);
    cb = plt.colorbar(im, cax=cax);
    title = ('rev 0' + str(rev) + ', LSD ' + str(LSD) + 
            f', {freq[0]:.2f}' + ' MHz')
    ax.set_title(title, fontsize=20)
    
    del ringmap, freq
    pass
        
    
#========================================================================

def events(observer, lsd):
    
    # Start and end times of the CSD
    st = observer.lsd_to_unix(lsd)
    et = observer.lsd_to_unix(lsd + 1)
    
    e = {}
    
    u2l = observer.unix_to_lsd
    
    t = observer.transit_times(eph["sun"], st, et)
    
    if len(t):
        e["sun_transit"] = u2l(t)[0]
    
    # Calculate the sun rise/set times on this sidereal day (it's not clear to me there
    # is exactly one of each per day, I think not (Richard))
    times, rises = observer.rise_set_times(eph["sun"], st, et, diameter=-1)
    for t, r in zip(times, rises):
        if r:
            e["sun_rise"] = u2l(t)
        else:
            e["sun_set"] = u2l(t)
    
    
    moon_time, moon_dec = observer.transit_times(eph["moon"], st, et, return_dec=True)
    
    if len(moon_time):
        e["lunar_transit"] = u2l(moon_time[0])
        e["lunar_dec"] = moon_dec[0]
    
    return e


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
    flag_mask = np.zeros_like(times, dtype=bool)

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
    
    # Calculate events like solar transit, rise ...
    ev = events(chime_obs, LSD)
    # Highlight the day time data
    sr = (ev["sun_rise"] % 1) * 360 if "sun_rise" in ev else 0
    ss = (ev["sun_set"] % 1) * 360 if "sun_set" in ev else 360
    
    if sr < ss:
        axis.axvspan(sr, ss, color="grey", alpha=0.5)
    else:
        axis.axvspan(0, ss, color="grey", alpha=0.5)
        axis.axvspan(sr, 360, color="grey", alpha=0.5)

    axis.axvline(sr, color="k", ls="--", lw=1)
    axis.axvline(ss, color="k", ls="--", lw=1)
        
    title = ('rev 0' + str(rev) + ', LSD ' + str(LSD))
    axis.set_title(title, fontsize=20)
    _ = axis.set_xticks(np.arange(0, 361, 45))

# ========================================================================

def plot_stability(
    rev, lsd, pol=None, min_dec=0.0, min_nfreq=100, norm_sigma=False, max_val=None,
    flag_daytime=True, flag_bad_data=True
):
    """Plot the variations in source spectra with respect to a template.

    Parameters
    ----------
    rev : int
        Revision of daily processing.
    lsd : int
        Local sidereal day.
    pol : list of str
        Polarisations.  Default is ["XX", "YY"].
    min_dec : float
        Only plot sources with a declination greater than
        this value in degrees.
    min_nfreq : float
        Only plot statistics calculated over frequency axis for sources with
        more than this number of unflagged frequencies.
    norm_sigma : bool
        If true, normalize by the standard deviation over days.
        If false, normalize by the absolute value of the template.
    max_val : float
        The maximum value of the color scale.  If not provided,
        then default to 3 (sigma) if norm_sigma is true and
        0.05 (fractional units) if norm_sigma is false.
    flag_daytime : bool
        Add a shaded region to the figures that indicates the daytime.
    flag_bad_data : bool
        Add a shaded region to the figures that indicates data that is bad
        according to a flag in the database.
    """

    if pol is None:
        pol = ["XX", "YY"]

    if max_val is None:
        max_val = 3.0 if norm_sigma else 0.05

    # Load the template
    patht = os.path.join(
        "/project/rpp-chime/chime/chime_processed/spectra",
        f"rev_{rev:02}",
        f"spectra_rev_{rev:02}_bright.h5",
    )

    template = containers.FormedBeam.from_file(patht)

    # Load the data for this sidereal day
    path = os.path.join(
        "/project/rpp-chime/chime/chime_processed/daily",
        f"rev_{rev:02}",
        f"{lsd}",
        f"sourceflux_lsd_{lsd}_bright.h5",
    )

    data = containers.FormedBeam.from_file(path)

    # Extract axes
    freq = data.freq

    ipol = np.array([list(data.pol).index(pstr) for pstr in pol])
    npol = ipol.size

    ra = data["position"]["ra"][:]
    dec = data["position"]["dec"][:]

    # Only consider sources above some minimum declination
    valid = np.flatnonzero(dec > min_dec)
    valid = valid[np.argsort(ra[valid])]

    ra = ra[valid]
    dec = dec[valid]

    # Flag bad data
    flag_rfi = ~rfi.frequency_mask(freq, timestamp=ephemeris.csd_to_unix(lsd))
    flag_time = ~_mask_flags(ephemeris.csd_to_unix(lsd + ra / 360.0), lsd)

    flag = (
        (data.weight[:] > 0.0)
        & (template.weight[:] > 0.0)
        & flag_rfi[np.newaxis, np.newaxis, :]
    )

    flag = flag[valid][:, ipol] & flag_time[:, np.newaxis, np.newaxis]

    # Calculate sunrise and sunset to indicate daytime data
    if flag_daytime:
        ev = events(chime_obs, lsd)
        sr = (ev["sun_rise"] % 1) * 360 if "sun_rise" in ev else 0
        ss = (ev["sun_set"] % 1) * 360 if "sun_set" in ev else 360

    # Query dataflags
    if flag_bad_data:
        bad_time_spans = flag_time_spans(lsd)
        bad_ra_woverlap = [[(max(chime_obs.unix_to_lsd(bts[1]), lsd) - lsd) * 360,
                            (min(chime_obs.unix_to_lsd(bts[2]), lsd+1) - lsd) * 360]
                           for bts in bad_time_spans]

        bad_ra_spans = []
        for begin, end in sorted(bad_ra_woverlap):
            if bad_ra_spans and bad_ra_spans[-1][1] >= begin:
                bad_ra_spans[-1][1] = max(bad_ra_spans[-1][1], end)
            else:
                bad_ra_spans.append([begin, end])

    # Calculate the fractional deviation in the source spectra
    ds = data.beam[:] - template.beam[:]

    if norm_sigma:
        lbl = r"$(S - S_{med}) \ / \ \sigma_{S}$"
        norm = np.sqrt(template.weight[:])
    else:
        lbl = r"$(S - S_{med}) \ / \ S_{med}$"
        norm = tools.invert_no_zero(np.abs(template.beam[:]))

    ds *= norm

    ds = ds[valid][:, ipol]

    # Calculate the median absolute fractional deviation over frequency for each source
    med_abs_ds = weighted_median.weighted_median(
        np.abs(ds).astype(np.float64), flag.astype(np.float64)
    )

    nfreq = np.sum(flag, axis=-1)
    med_abs_ds = np.where(nfreq > min_nfreq, med_abs_ds, np.nan)

    # Define plot parameters
    ra_grid = np.arange(0.0, 360.0, 1.0)
    index = _search_nearest(ra, ra_grid)

    extent = (freq[-1], freq[0], ra_grid[0], ra_grid[-1])
    cmap = plt.get_cmap("coolwarm")

    # Create plot
    fig = plt.figure(num=1, figsize=(30, 7.5 * npol))

    gspec = matplotlib.gridspec.GridSpec(
        npol, 2, width_ratios=[1, 5], wspace=0.04, hspace=0.05
    )

    alpha = 0.25
    color_bad = "grey"
    color_day="gold"

    # Loop over the requested polarisations
    for pp, pstr in enumerate(pol):
        # Plot the median (over frequency) fractional deviation
        # as a function of source RA
        plt.subplot(gspec[pp, 0])
        plt.plot(
            med_abs_ds[index, pp], ra_grid, color="k", linestyle="-", marker="None"
        )

        if flag_daytime:
            if sr < ss:
                plt.axhspan(sr, ss, color=color_day, alpha=alpha)
            else:
                plt.axhspan(0, ss, color=color_day, alpha=alpha)
                plt.axhspan(sr, 360, color=color_day, alpha=alpha)

        if flag_bad_data:
            for brs in bad_ra_spans:
                plt.axhspan(brs[0], brs[1], color=color_bad, alpha=alpha)

        plt.xlim([0.0, max_val])
        plt.ylim(ra_grid[0], ra_grid[-1])
        plt.grid()

        plt.ylabel("RA [deg]")
        if pp < (npol - 1):
            plt.gca().axes.get_xaxis().set_ticklabels([])
        else:
            plt.xlabel(r"med$_{\nu}(|$" + lbl + r"$|)$")

        plt.text(
            0.95,
            0.95,
            f"Pol {pstr}",
            color="red",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            horizontalalignment="right",
        )

        # Create an image of the fractional deviation
        # as a function of frequency and source RA
        plt.subplot(gspec[pp, 1])
        mplot = np.where(flag[index, pp], ds[index, pp], np.nan)

        img = plt.imshow(
            mplot[:, ::-1],
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=extent,
            cmap=cmap,
            vmin=-max_val,
            vmax=max_val,
        )

        cbar = plt.colorbar(img)
        cbar.set_label(lbl)

        if flag_daytime:
            if sr < ss:
                plt.axhspan(sr, ss, color=color_day, alpha=alpha)
            else:
                plt.axhspan(0, ss, color=color_day, alpha=alpha)
                plt.axhspan(sr, 360, color=color_day, alpha=alpha)

        if flag_bad_data:
            for brs in bad_ra_spans:
                plt.axhspan(brs[0], brs[1], color=color_bad, alpha=alpha)

        plt.ylim(extent[2], extent[3])

        plt.gca().axes.get_yaxis().set_ticklabels([])
        if pp < (npol - 1):
            plt.gca().axes.get_xaxis().set_ticklabels([])
        else:
            plt.xlabel("Frequency [MHz]")

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
            
#========================================================================

def plotRM_tempSub(rev, LSD, fi = 400, pi = 3):
    """
    Plots the template subtracted ringmap for a given LSD with additional
    details a) flagged out time ranges at the top, b) Sensitivity plot at
    the bottom
    
    Parameters
    ----------
    rev : int
          Revision number
    LSD : int
          Day number
    fi  : freq index
    pi  : pol index 
    
    Returns
    -------
    template subtracted ringmap
    """
    
    path = "/project/rpp-chime/chime/chime_processed/daily/"
    path += "rev_0" + str(rev) + "/" + str(LSD) + "/"
    
    # load ringmap
    data = "ringmap_lsd_" + str(LSD) + ".zarr.zip" 
    ringmap = containers.RingMap.from_file(
            os.path.join(path, data), freq_sel=slice(fi, fi+1), pol_sel=slice(pi, pi + 1))
    csd_arr = LSD + ringmap.index_map["ra"][:] / 360.0
    
    rm = ringmap.map[0, 0, 0]
    rm_weight_agg = ringmap.weight[0, 0].mean(axis=-1)
    freq = ringmap.freq
    weight_mask = (rm_weight_agg == 0.0)
    
    
    #Calculate the sensitivity plot to display. Take the ratio of the measured and expected radiometer variances and flatten the frequencies
    
    data = "sensitivity_lsd_" + str(LSD) + ".h5"
    sensitivity = containers.SystemSensitivity.from_file(
        os.path.join(path, data))
    
    data = "rfi_mask_lsd_" + str(LSD) + ".h5"
    rfi = containers.RFIMask.from_file(os.path.join(path, data))
    
    sp = 1
    sens_arr = sensitivity.measured[:, 1]


    rfi_arr = rfi.mask[:]
    sens_csd = ephemeris.csd(sensitivity.time)
    sens_csd = ephemeris.csd(sensitivity.time)

    sensrat = (sensitivity.measured[:, sp] *
               tools.invert_no_zero(sensitivity.radiometer[:, sp]))
    sensrat /= np.median(sensrat, axis=1)[:, np.newaxis]
    sensrat *= np.where(rfi_arr == 0, 1, np.nan)
    
        
    # calculate a mask for the ringmap
    topos = sf_obs.vector_functions[-1]    
    sf_times = ctime.unix_to_skyfield_time(chime_obs.lsd_to_unix(csd_arr.ravel()))   
    daytime = almanac.sunrise_sunset(eph, topos)(sf_times).reshape(csd_arr.shape)
    flag_mask = np.zeros_like(csd_arr, dtype=bool)
    
    # Calculate the set of flags for this day
    flags_by_type = {
        #"Daytime": daytime[di],
        "Weights": weight_mask,
    }

    u2l = chime_obs.unix_to_lsd
    
    for type_, ua, ub in flag_time_spans(LSD):
        ca = u2l(ua)
        cb = u2l(ub)
        
        flag_mask[(csd_arr > ca) & (csd_arr < cb)] = True
        
        if (ca > LSD + 1) or cb < LSD:
            continue

        if type_ not in flags_by_type:
            flags_by_type[type_] = np.zeros_like(csd_arr, dtype=bool)

        flags_by_type[type_][(csd_arr > ca) & (csd_arr < cb)] = True        
    
    rm_masked = np.where((daytime | flag_mask | weight_mask)[:, np.newaxis], 
                         np.nan, rm)
    rm_masked_all = np.where((flag_mask | weight_mask)[:, np.newaxis], np.nan, rm)

    # load ringmap template
    path_stack = "/project/rpp-chime/chime/chime_processed/stacks/"
    path_stack += "rev_0" + str(rev) + "/test0/all/ringmap.h5"
    rm_stack = containers.RingMap.from_file(path_stack, freq_sel=slice(fi, fi + 1),
                                            pol_sel=slice(pi, pi + 1))
    rm_stack = rm_stack.map[0, 0, 0]
    ## Need to match the elevations being used. This is very crude. We should actually just generate a new ringmap with a match elevation axis (Richard)
    rm_stack = rm_stack[..., ::2]
    
    # NOTE: do a very straightforward template subtraction and destriping
    ra = ringmap.index_map["ra"][:]
    md = rm_masked_all - rm_stack
    md -= np.nanmedian(md, axis=0)
    
    # Calculate events like solar transit, rise ...
    ev = events(chime_obs, LSD)
        
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 13), gridspec_kw=dict(height_ratios=[1, 10], hspace=0.0))
    
    fontsize = 20
    labelsize = 20

    # Plot the flagged out time ranges at the very top
    for ii, (type_, series) in enumerate(flags_by_type.items()):
        axes[0].fill_between(ra, ii, ii + 1, where=series, label=type_, color=f"C{ii}", alpha=0.5)
    axes[0].legend()
    axes[0].set_yticks([])
    axes[0].set_ylim(0, ii + 1)


    # Plot the template subtracted ringmap
    vl = 5
    cmap = copy.copy(matplotlib.cm.inferno)
    cmap.set_bad("grey")
    im = axes[1].imshow(md.T, vmin=-vl, vmax=vl, aspect="auto", extent=(0, 360, -1, 1), origin="lower", cmap=cmap)
    axes[1].set_yticks([-1, -0.5, 0, 0.5, 1])
    axes[1].yaxis.set_tick_params(labelsize = labelsize)
    axes[1].xaxis.set_tick_params(labelsize = labelsize)
    axes[1].set_ylabel("sin(ZA)", fontsize = fontsize)
    axes[1].set_xlabel("RA [degrees]", fontsize = fontsize)
    cb = plt.colorbar(im, aspect=50, orientation='horizontal', pad =0.1);

    # Put a ring around the location of the moon if it transits on this day
    if "lunar_transit" in ev:
        lunar_ra = (ev["lunar_transit"] % 1)* 360.0
        lunar_za = np.sin(np.radians(ev["lunar_dec"] - 49.0))
        circle(axes[1], lunar_ra, lunar_za, radius=0.2, facecolor="none", edgecolor="k")

    # Highlight the day time data
    sr = (ev["sun_rise"] % 1) * 360 if "sun_rise" in ev else 0
    ss = (ev["sun_set"] % 1) * 360 if "sun_set" in ev else 360
    for ax in axes[1:]:
        if sr < ss:
            ax.axvspan(sr, ss, color="grey", alpha=0.5)
        else:
            ax.axvspan(0, ss, color="grey", alpha=0.5)
            ax.axvspan(sr, 360, color="grey", alpha=0.5)

        ax.axvline(sr, color="k", ls="--", lw=1)
        ax.axvline(ss, color="k", ls="--", lw=1)

    
    # Give the overall plot a title identifying the CSD
    title = 'rev 0' + str(rev) + ', LSD ' + str(LSD) + f', {freq[0]:.2f}' + ' MHz'
    axes[0].set_title(title, fontsize = fontsize)

