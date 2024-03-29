import copy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from skyfield import almanac

from datetime import datetime

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


# ==== Locations and helper functions for loading files ====

base_path = Path("/project/rpp-chime/chime/chime_processed/daily")
template_path = Path("/project/rpp-chime/chime/validation/templates")

_file_spec = {
    "ringmap": ("ringmap_", ".zarr.zip"),
    "delayspectrum": ("delayspectrum_", ".h5"),
    "delayspectrum_hpf": ("delayspectrum_hpf_", ".h5"),
    "sensitivity": ("sensitivity_", ".h5"),
    "rfi_mask": ("rfi_mask_", ".h5"),
    "sourceflux": ("sourceflux_", "_bright.h5"),
}


def _get_rev_path(type_: str, rev: int, lsd: int) -> Path:
    if type_ not in _file_spec:
        raise ValueError(f"Unknown file type {type_}.")

    prefix, suffix = _file_spec[type_]

    return base_path / f"rev_{rev:02d}" / f"{lsd:d}" / f"{prefix}lsd_{lsd:d}{suffix}"


def get_csd(day: int | str = None, num_days: int = 0, lag: int = 0) -> int:
    """Get a csd from an integer or a string with format yyyy/mm/dd.

    If None, return the current CSD.
    """
    if day is None:
        return int(ephemeris.chime.get_current_lsd() - num_days - lag)

    if isinstance(day, str):
        day = datetime.strptime(day, "%Y/%m/%d").timestamp()
        return int(ephemeris.unix_to_csd(day))

    return int(day)


# ==========================================================================


def _mask_baselines(baseline_vec, single_mask=False):
    """Mask long baselines in a delay spectrum."""

    bl_mask = np.zeros((4, baseline_vec.shape[0]), dtype=bool)
    bl_mask[0] = baseline_vec[:, 0] < 10
    bl_mask[1] = (baseline_vec[:, 0] > 10) & (baseline_vec[:, 0] < 30)
    bl_mask[2] = (baseline_vec[:, 0] > 30) & (baseline_vec[:, 0] < 50)
    bl_mask[3] = baseline_vec[:, 0] > 50

    if single_mask:
        bl_mask = np.any(bl_mask, axis=0)

    return bl_mask


def _hide_axis(ax):
    """Hide axis ticks and frame without removing axis."""

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(left=False, bottom=False)


def plotDS(rev, LSD, hpf=False, clim=[1e-3, 1e2], cmap="inferno"):
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
    clim : list, optional (default [1e-3, 1e2])
        min, max values in the colorscale
    cmap : colormap, optional (default 'inferno')

    Returns
    -------
    Delay Spectrum
    """

    type_ = "delayspectrum_hpf" if hpf else "delayspectrum"
    path = _get_rev_path(type_, rev, LSD)

    DS = containers.DelaySpectrum.from_file(path)

    tau = DS.index_map["delay"] * 1e3
    DS_Spec = DS.spectrum

    baseline_vec = DS.index_map["baseline"]
    bl_mask = _mask_baselines(baseline_vec)

    fig, ax = plt.subplots(
        1, 4, figsize=(15, 8), sharey=True, gridspec_kw={"width_ratios": [1, 2, 2, 2]}
    )
    imshow_params = {
        "origin": "lower",
        "aspect": "auto",
        "interpolation": "none",
        "norm": LogNorm(),
        "clim": clim,
        "cmap": cmap,
    }

    for i in range(4):
        baseline_idx_sorted = baseline_vec[bl_mask[i], 1].argsort()
        extent = [
            baseline_vec[bl_mask[i], 1].min(),
            baseline_vec[bl_mask[i], 1].max(),
            tau[0],
            tau[-1],
        ]
        im = ax[i].imshow(
            DS_Spec[bl_mask[i]][baseline_idx_sorted].T.real,
            extent=extent,
            **imshow_params,
        )
        ax[i].xaxis.set_tick_params(labelsize=18)
        ax[i].yaxis.set_tick_params(labelsize=18)
        ax[i].set_title(f"{i}-cyl", fontsize=20)

    # ax[0].set_ylabel("Delay [ns]")
    fig.supxlabel("NS baseline length [m]", fontsize=20)
    fig.supylabel("Delay [ns]", fontsize=20)
    title = "rev 0" + str(rev) + ", CSD " + str(LSD) + ", hpf = " + str(hpf)
    fig.suptitle(title, fontsize=20)
    fig.subplots_adjust(wspace=0.05)
    fig.colorbar(
        im, ax=ax, orientation="vertical", label="Signal Power", pad=0.02, aspect=40
    )

    del DS
    pass


def plotMultipleDS(
    rev,
    csd_start,
    num_days,
    view="grid",
    reverse=True,
    hpf=False,
    clim=[1e-3, 1e2],
    cmap="inferno",
):
    """Plot multiple delay spectra in a given range.

    Parameters
    ----------
    rev : int
        Revision number
    csd_start : int
        First csd in the range
    num_days : int
        Number of days to plot, starting at `csd_start`
    view : str, optional (default "grid")
        How to display the plots. If `grid`, delay spectra are shown
        on a dense nx3 grid. If `list`, delay spectra are plotted one
        at a time using the plotting format in `plotDS`
    reverse : bool, optional (default True)
        If true, display days in decreasing order
    hpf : bool, optional (default False)
          with/without high pass filter (True/False)
    clim : list, optional (default [1e-3, 1e2])
        min, max values in the colorscale
    cmap : colormap, optional (default 'inferno')
    """

    if num_days < 1:
        # Why would we want this
        print("No days requested")
        return

    if view not in {"list", "grid"}:
        print("Invalid value for kwarg `view`. Using default `grid`")
        view = "grid"

    # Accumulate the number of days available
    count = 0
    csds = list(range(csd_start, csd_start + num_days))
    if reverse:
        csds = csds[::-1]

    if view == "list":
        for csd in csds:
            try:
                plotDS(rev, csd, hpf, clim, cmap)
            except FileNotFoundError:
                count += 1

        print(f"Data products found for {num_days - count}/{num_days} days.")

        return

    type_ = "delayspectrum_hpf" if hpf else "delayspectrum"

    # Sort out the grid shape
    if not bool(num_days % 2):
        # Day number is divisible by 2
        plt_shape_ = (num_days // 2, 2)
    else:
        # Otherwise use a 3x3 grid
        extra_row = int(bool(num_days % 3))
        plt_shape_ = (num_days // 3 + extra_row, 3)

    # Set up a grid
    extra_row = int(num_days % 3 != 0)
    fig, ax = plt.subplots(
        *plt_shape_,
        figsize=(int(10 * plt_shape_[1]), int(10 * plt_shape_[0])),
        sharey=True,
        sharex=True,
        layout="constrained",
    )
    # The is for consistency of indexing axes
    ax = np.atleast_2d(ax)
    # If no data is plotted, we probably shouldn't display anything
    im = None
    imshow_params = {
        "origin": "lower",
        "aspect": "auto",
        "interpolation": "nearest",
        "norm": LogNorm(),
        "clim": clim,
        "cmap": cmap,
    }

    for i, csd in enumerate(csds):
        ax_row = i // plt_shape_[1]
        ax_col = i % plt_shape_[1]

        path = _get_rev_path(type_, rev, csd)

        try:
            DS = containers.DelaySpectrum.from_file(path)
        except FileNotFoundError:
            # Hide this axis, but don't actually disable it
            _hide_axis(ax[ax_row, ax_col])
            # grey out this subplot
            ax[ax_row, ax_col].set_facecolor("#686868")
            count += 1
            continue

        # Get the axis extent and any masking
        tau = DS.index_map["delay"] * 1e3
        baseline_vec = DS.index_map["baseline"]
        bl_mask = _mask_baselines(baseline_vec, single_mask=True)
        bl_mask = np.tile(bl_mask, (len(tau), 1))

        extent = [0, baseline_vec.shape[0], tau[0], tau[-1]]

        im = ax[ax_row, ax_col].imshow(
            np.ma.masked_array(DS.spectrum[:].T.real, mask=~bl_mask.T),
            extent=extent,
            **imshow_params,
        )
        date = ephemeris.csd_to_unix(int(csd))
        date = datetime.utcfromtimestamp(date).strftime("%Y-%m-%d")
        ax[ax_row, ax_col].set_title(f"{csd} ({date})")

    if im is None:
        # We never actually plotted any data
        print("No data available in this range")
        del fig
        return

    fig.colorbar(im, ax=ax, location="top", aspect=40, pad=0.01)
    fig.supxlabel("NS baseline", fontsize=40)
    fig.supylabel("Delay [ns]", fontsize=40)
    title = f"Signal Power - rev {rev:02d}, CSD range {csd_start}-{csd_start+num_days-1}, hpf = {hpf}"
    fig.suptitle(title, fontsize=40)

    # Remove the extra unused subplots
    for i in range(ax.size - num_days):
        _hide_axis(ax[-1, -i - 1])

    # set the axis labelsize everywhere
    for _ax in ax.flatten():
        _ax.xaxis.set_tick_params(labelsize=18)
        _ax.yaxis.set_tick_params(labelsize=18)

    del DS
    print(f"Data products found for {num_days - count}/{num_days} days.")


# ========================================================================


def plotRingmap(rev, LSD, vmin=-5, vmax=20, fi=400, flag_mask=True):
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

    path = _get_rev_path("ringmap", rev, LSD)

    ringmap = containers.RingMap.from_file(path, freq_sel=slice(fi, fi + 1))

    freq = ringmap.freq
    m = ringmap.map[0, 0, 0].T[::-1]
    w = ringmap.weight[0, 0].T[::-1]

    nanmask = np.where(w == 0, np.nan, 1)
    nanmask *= np.where(
        _mask_flags(ephemeris.csd_to_unix(LSD + ringmap.ra / 360.0), LSD), np.nan, 1
    )

    m -= np.nanmedian(m * nanmask, axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    cmap = copy.copy(matplotlib.cm.inferno)
    cmap.set_bad("grey")

    if flag_mask:
        im = ax.imshow(
            m * nanmask,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            extent=(0, 360, -1, 1),
            cmap=cmap,
        )
    else:
        im = ax.imshow(
            m, vmin=vmin, vmax=vmax, aspect="auto", extent=(0, 360, -1, 1), cmap=cmap
        )
    ax.set_xlabel("RA [degrees]")
    ax.set_ylabel("sin(ZA)")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1.5%", pad=0.25)
    fig.colorbar(im, cax=cax)
    title = "rev 0" + str(rev) + ", LSD " + str(LSD) + f", {freq[0]:.2f}" + " MHz"
    ax.set_title(title, fontsize=20)


# ========================================================================


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
        "bad_calibration_fpga_restart",
        #'globalflag',
        #'acjump',
        "acjump_sd",
        #'rain',
        #'rain_sd',
        "bad_calibration_acquisition_restart",
        #'misc',
        #'rain1mm',
        "rain1mm_sd",
        "srs/bad_ringmap_broadband",
        "bad_calibration_gains",
        "snow",
        "decorrelated_cylinder",
    ]

    flags = (
        df.DataFlag.select()
        .where(df.DataFlag.start_time < ut_end, df.DataFlag.finish_time > ut_start)
        .join(df.DataFlagType)
        .where(df.DataFlagType.name << bad_flags)
    )

    flag_time_spans = [(f.type.name, f.start_time, f.finish_time) for f in flags]

    return flag_time_spans


def _mask_flags(times, LSD):
    flag_mask = np.zeros_like(times, dtype=bool)

    for type_, ca, cb in flag_time_spans(LSD):
        flag_mask[(times > ca) & (times < cb)] = True

    return flag_mask


def plotSens(rev, LSD, vmin=0.995, vmax=1.005):
    path = _get_rev_path("sensitivity", rev, LSD)
    sens = containers.SystemSensitivity.from_file(path)

    rfi_path = _get_rev_path("rfi_mask", rev, LSD)
    rfm = containers.RFIMask.from_file(rfi_path)

    sp = 0
    fig, axis = plt.subplots(1, 1, figsize=(15, 10))

    sensrat = sens.measured[:, sp] * tools.invert_no_zero(sens.radiometer[:, sp])
    sensrat /= np.median(sensrat, axis=1)[:, np.newaxis]
    sensrat *= np.where(rfm.mask[:] == 0, 1, np.nan)
    sensrat *= np.where(_mask_flags(sens.time, LSD), np.nan, 1)

    cmap = copy.copy(matplotlib.cm.viridis)
    cmap.set_bad("#aaaaaa")

    im = axis.imshow(
        sensrat,
        extent=(0, 360, 400, 800),
        cmap=cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="1.5%", pad=0.25)
    fig.colorbar(im, cax=cax)
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

    title = "rev 0" + str(rev) + ", LSD " + str(LSD)
    axis.set_title(title, fontsize=20)
    _ = axis.set_xticks(np.arange(0, 361, 45))


# ========================================================================


def plot_stability(
    rev,
    lsd,
    pol=None,
    min_dec=0.0,
    min_nfreq=100,
    norm_sigma=False,
    max_val=None,
    flag_daytime=True,
    flag_bad_data=True,
    template_rev=6,
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
    template_rev : int
        The revision to use for the template.
    """

    if pol is None:
        pol = ["XX", "YY"]

    if max_val is None:
        max_val = 3.0 if norm_sigma else 0.05

    # Load the template
    patht = template_path / f"spectra_rev{template_rev:02d}.h5"
    template = containers.FormedBeam.from_file(patht)

    # Load the data for this sidereal day
    path = _get_rev_path("sourceflux", rev, lsd)
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
        bad_ra_woverlap = [
            [
                (max(chime_obs.unix_to_lsd(bts[1]), lsd) - lsd) * 360,
                (min(chime_obs.unix_to_lsd(bts[2]), lsd + 1) - lsd) * 360,
            ]
            for bts in bad_time_spans
        ]

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
    fig, axs = plt.subplots(
        npol,
        2,
        figsize=(30, 7.5 * npol),
        gridspec_kw=dict(width_ratios=[1, 5], wspace=0.04, hspace=0.05),
    )

    alpha = 0.25
    color_bad = "grey"
    color_day = "gold"

    # Loop over the requested polarisations
    for pp, pstr in enumerate(pol):
        # Plot the median (over frequency) fractional deviation
        # as a function of source RA
        ax = axs[pp, 0]
        ax.plot(med_abs_ds[index, pp], ra_grid, color="k", linestyle="-", marker="None")

        if flag_daytime:
            if sr < ss:
                ax.axhspan(sr, ss, color=color_day, alpha=alpha)
            else:
                ax.axhspan(0, ss, color=color_day, alpha=alpha)
                ax.axhspan(sr, 360, color=color_day, alpha=alpha)

        if flag_bad_data:
            for brs in bad_ra_spans:
                ax.axhspan(brs[0], brs[1], color=color_bad, alpha=alpha)

        ax.set_xlim(0.0, max_val)
        ax.set_ylim(ra_grid[0], ra_grid[-1])
        ax.grid()

        ax.set_ylabel("RA [deg]")
        if pp < (npol - 1):
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r"med$_{\nu}(|$" + lbl + r"$|)$")

        ax.text(
            0.95,
            0.95,
            f"Pol {pstr}",
            color="red",
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
        )

        # Create an image of the fractional deviation
        # as a function of frequency and source RA
        ax = axs[pp, 1]
        mplot = np.where(flag[index, pp], ds[index, pp], np.nan)

        img = ax.imshow(
            mplot[:, ::-1],
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=extent,
            cmap=cmap,
            vmin=-max_val,
            vmax=max_val,
        )

        cbar = fig.colorbar(img)
        cbar.set_label(lbl)

        if flag_daytime:
            if sr < ss:
                ax.axhspan(sr, ss, color=color_day, alpha=alpha)
            else:
                ax.axhspan(0, ss, color=color_day, alpha=alpha)
                ax.axhspan(sr, 360, color=color_day, alpha=alpha)

        if flag_bad_data:
            for brs in bad_ra_spans:
                ax.axhspan(brs[0], brs[1], color=color_bad, alpha=alpha)

        ax.set_ylim(extent[2], extent[3])

        ax.set_yticklabels([])
        if pp < (npol - 1):
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Frequency [MHz]")


# ========================================================================


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
    trans = fig.dpi_scale_trans + matplotlib.transforms.ScaledTranslation(
        x, y, ax.transData
    )
    circle = patches.Circle((0, 0), radius, transform=trans, **kwargs)

    # Draw circle
    return ax.add_artist(circle)


# ========================================================================


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
                ranges.append((last_cut, ii + 1))
                last_cut = ii + 1

        ranges.append((last_cut, len(ax)))

        return ranges

    artists = []
    for xs, xe in _find_splits(x):
        for ys, ye in _find_splits(y):
            xa = x[xs:xe]
            ya = y[ys:ye]
            ca = c[ys:ye, xs:xe]

            artists.append(
                axis.imshow(ca, extent=(xa[0], xa[-1], ya[-1], ya[0]), *args, **kwargs)
            )

    return artists


# ========================================================================


def plotRM_tempSub(rev, LSD, fi=400, pi=3, daytime=False, template_rev=3):
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
    daytime : bool
        Highlight the day time period or not.
    template_rev: int
        The revision to use for the background template.

    Returns
    -------
    template subtracted ringmap
    """

    # load ringmap
    path = _get_rev_path("ringmap", rev, LSD)
    ringmap = containers.RingMap.from_file(
        path, freq_sel=slice(fi, fi + 1), pol_sel=slice(pi, pi + 1)
    )
    csd_arr = LSD + ringmap.index_map["ra"][:] / 360.0

    rm = ringmap.map[0, 0, 0]
    rm_weight_agg = ringmap.weight[0, 0].mean(axis=-1)
    freq = ringmap.freq
    weight_mask = rm_weight_agg == 0.0

    # calculate a mask for the ringmap
    topos = sf_obs.vector_functions[-1]
    sf_times = ctime.unix_to_skyfield_time(chime_obs.lsd_to_unix(csd_arr.ravel()))
    daytime_mask = almanac.sunrise_sunset(eph, topos)(sf_times).reshape(csd_arr.shape)
    flag_mask = np.zeros_like(csd_arr, dtype=bool)

    # Calculate the set of flags for this day
    flags_by_type = {
        "Weights": weight_mask,
    }

    if daytime:
        flags_by_type["Daytime"] = daytime_mask

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

    rm_masked_all = np.where((flag_mask | weight_mask)[:, np.newaxis], np.nan, rm)

    # load ringmap template
    path_stack = template_path / f"ringmap_rev{template_rev:02d}.zarr.zip"
    rm_stack = containers.RingMap.from_file(
        path_stack, freq_sel=slice(fi, fi + 1), pol_sel=slice(pi, pi + 1)
    )
    rm_stack = rm_stack.map[0, 0, 0]

    # NOTE: do a very straightforward template subtraction and destriping
    ra = ringmap.index_map["ra"][:]
    md = rm_masked_all - rm_stack
    md -= np.nanmedian(md, axis=0)

    # Calculate events like solar transit, rise ...
    ev = events(chime_obs, LSD)

    _, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(14, 13),
        gridspec_kw=dict(height_ratios=[1, 10], hspace=0.0),
    )

    fontsize = 20
    labelsize = 20

    # Plot the flagged out time ranges at the very top
    for ii, (type_, series) in enumerate(flags_by_type.items()):
        axes[0].fill_between(
            ra, ii, ii + 1, where=series, label=type_, color=f"C{ii}", alpha=0.5
        )
    axes[0].legend()
    axes[0].set_yticks([])
    axes[0].set_ylim(0, ii + 1)

    # Plot the template subtracted ringmap
    vl = 5
    cmap = copy.copy(matplotlib.cm.inferno)
    cmap.set_bad("grey")
    im = axes[1].imshow(
        md.T,
        vmin=-vl,
        vmax=vl,
        aspect="auto",
        extent=(0, 360, -1, 1),
        origin="lower",
        cmap=cmap,
    )
    axes[1].set_yticks([-1, -0.5, 0, 0.5, 1])
    axes[1].yaxis.set_tick_params(labelsize=labelsize)
    axes[1].xaxis.set_tick_params(labelsize=labelsize)
    axes[1].set_ylabel("sin(ZA)", fontsize=fontsize)
    axes[1].set_xlabel("RA [degrees]", fontsize=fontsize)
    cb = plt.colorbar(im, aspect=50, orientation="horizontal", pad=0.1)

    # Put a ring around the location of the moon if it transits on this day
    if "lunar_transit" in ev:
        lunar_ra = (ev["lunar_transit"] % 1) * 360.0
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
    title = "rev 0" + str(rev) + ", LSD " + str(LSD) + f", {freq[0]:.2f}" + " MHz"
    axes[0].set_title(title, fontsize=fontsize)
