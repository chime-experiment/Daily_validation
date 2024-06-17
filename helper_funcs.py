import copy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from skyfield import almanac

from datetime import datetime

import logging

logger = logging.getLogger(__name__)

from caput import weighted_median
from caput.tools import invert_no_zero
from caput import time as ctime
from draco.core import containers
from draco.util import tools
from draco.analysis.sidereal import _search_nearest
from ch_util import cal_utils, ephemeris, rfi
from chimedb import core, dataflag as df

eph = ephemeris.skyfield_wrapper.ephemeris
chime_obs = ephemeris.chime
sf_obs = chime_obs.skyfield_obs()


# ==== Plot color defaults ====
_BAD_VALUE_COLOR = "#1a1a1a"
_SOURCES = ["sun", "moon", "CAS_A", "CYG_A", "TAU_A", "VIR_A", "B0329+54"]

# ==== Locations and helper functions for loading files ====

base_path = Path("/project/rpp-chime/chime/chime_processed/daily")
template_path = Path("/project/rpp-chime/chime/validation/templates")

_file_spec = {
    "ringmap": ("ringmap_", ".zarr.zip"),
    "delayspectrum": ("delayspectrum_", ".h5"),
    "delayspectrum_hpf": ("delayspectrum_hpf_", ".h5"),
    "sensitivity": ("sensitivity_", ".h5"),
    "chisq": ("chisq_", ".h5"),
    "power": ("lowpass_power_2cyl_", ".h5"),
    "chisq_mask": ("rfi_mask_chisq_", ".h5"),
    "stokesi_mask": ("rfi_mask_stokesi_", ".h5"),
    "sens_mask": ("rfi_mask_sensitivity_", ".h5"),
    "fact_mask": ("rfi_mask_factorized_", ".h5"),
    "sourceflux": ("sourceflux_", "_bright.h5"),
}


def _fail_quietly(func):
    """Just log any exceptions."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as err:
            logger.debug(f"Function {func.__name__} failed with error:\n{err}")
            return None

    return wrapper


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


def _format_title(rev, LSD):
    """Return a title string for plots."""
    return f"rev_{int(rev):02d}, CSD {int(LSD):04d}"


def _select_CSD_bounds(times, LSD, obs=chime_obs):
    """Return indices representing the times found within the LSD"""
    ra = obs.unix_to_lsd(times) - LSD

    return (ra >= 0.0) & (ra < 1.0)


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


@_fail_quietly
def plotDS(
    rev,
    LSD,
    hpf=False,
    clim=[[1e-4, 1e2], [1e-4, 1e-2]],
    cmap="inferno",
    dynamic_clim=False,
):
    """
    Plots the delay spectrum for a given LSD.

    Show the delay spectrum in two different color ranges.

    Parameters
    ----------
    rev : int
          Revision number
    LSD : int
          Day number
    hpf : bool, optional (default False)
          with/without high pass filter (True/False)
    clim : list[list], optional (default [[1e-3, 1e2], [1e-3, 1e-2]])
        min, max values in the colorscale for each plot
    cmap : colormap, optional (default 'inferno')
    dynamic_clim : bool, optional
        If true, clim will be adjusted to try to compress
        the upper limit without saturating
    """

    type_ = "delayspectrum_hpf" if hpf else "delayspectrum"
    path = _get_rev_path(type_, rev, LSD)

    DS = containers.DelaySpectrum.from_file(path)

    tau = DS.index_map["delay"] * 1e3
    DS_Spec = DS.spectrum

    if dynamic_clim:
        # Order of magnitude of the mean of the 2 cyl sep high-delay region
        # This is fairly arbitrary and not very robust, so it should be
        # improved
        # Only modify the second clims
        spec_m = np.floor(np.log10(np.median(DS_Spec)))
        delta = np.floor(np.log10(clim[1][0])) - spec_m
        clim[1][1] = 10 ** (np.floor(np.log10(clim[1][1])) - delta)

    baseline_vec = DS.index_map["baseline"]
    bl_mask = _mask_baselines(baseline_vec)

    # Make the master figure
    mfig = plt.figure(layout="constrained", figsize=(35, 12))
    # Make the two sub-figures
    subfigs = mfig.subfigures(1, 2, wspace=0.1)

    for ii, fig in enumerate(subfigs):
        ax = fig.subplots(1, 4, sharey=True, gridspec_kw={"width_ratios": [1, 2, 2, 2]})

        imshow_params = {
            "origin": "lower",
            "aspect": "auto",
            "interpolation": "nearest",
            "norm": LogNorm(vmin=clim[ii][0], vmax=clim[ii][1]),
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

        fig.supxlabel("NS baseline length [m]", fontsize=20)
        fig.supylabel("Delay [ns]", fontsize=20)
        title = _format_title(rev, LSD) + ", hpf = " + str(hpf)
        fig.suptitle(title, fontsize=20)
        fig.colorbar(
            im, ax=ax, orientation="vertical", label="Signal Power", pad=0.02, aspect=40
        )

    plt.show()


@_fail_quietly
def plotMultipleDS(
    rev,
    csd_start,
    num_days,
    reverse=True,
    hpf=False,
    clim=[1e-4, 1e0],
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

    # Accumulate the number of days available
    count = 0
    csds = list(range(csd_start, csd_start + num_days))
    if reverse:
        csds = csds[::-1]

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


@_fail_quietly
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
            interpolation="nearest",
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
    title = _format_title(rev, LSD) + f", {freq[0]:.2f}" + " MHz"
    ax.set_title(title, fontsize=20)


# ========================================================================


def events(observer, lsd):
    # Start and end times of the CSD
    st = observer.lsd_to_unix(lsd)
    et = observer.lsd_to_unix(lsd + 1)

    e = {}

    u2l = observer.unix_to_lsd

    sources = [src for src in _SOURCES if src not in {"sun", "moon"}]

    bodies = {src: ephemeris.source_dictionary[src] for src in sources}
    bodies["moon"] = eph["moon"]

    # Sun is handled differently because we care about rise/set
    # rather than just transit

    tt = observer.transit_times(eph["sun"], st, et)

    if tt:
        e["sun_transit"] = u2l(tt[0])

    # Calculate the sun rise/set times on this sidereal day (it's not clear to me there
    # is exactly one of each per day, I think not (Richard))
    times, rises = observer.rise_set_times(eph["sun"], st, et, diameter=-1)
    for t, r in zip(times, rises):
        if r:
            e["sun_rise"] = u2l(t)
        else:
            e["sun_set"] = u2l(t)

    for name, body in bodies.items():

        tt = observer.transit_times(body, st, et)

        if tt:
            tt = tt[0]
        else:
            continue

        sf_time = ephemeris.unix_to_skyfield_time(tt)
        pos = observer.skyfield_obs().at(sf_time).observe(body)

        alt = pos.apparent().altaz()[0]
        dec = pos.cirs_radec(sf_time)[1]

        e[f"{name}_dec"] = dec.degrees

        # Make sure body is above the horizon
        if alt.radians > 0.0:
            # Estimate the amount of time that the body is in the primary
            # beam to 2 sigma
            window_deg = 2.0 * cal_utils.guess_fwhm(
                800.0, pol="X", dec=dec.radians, sigma=True
            )
            window_sec = window_deg * 240.0 * ephemeris.SIDEREAL_S

            # Enter the transit timings in the output dict
            e[f"{name}_transit"] = u2l(tt)
            e[f"{name}_transit_start"] = u2l(tt - window_sec)
            e[f"{name}_transit_end"] = u2l(tt + window_sec)

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


def _highlight_sources(LSD, axobj, sources=_SOURCES, obs=chime_obs):
    """Add shaded regions over source objects."""
    ev = events(obs, LSD)

    for src in sources:
        if src == "sun":
            start = (ev[f"{src}_rise"] % 1) * 360.0
            finish = (ev[f"{src}_set"] % 1) * 360.0
        else:
            start = (ev[f"{src}_transit_start"] % 1) * 360.0
            finish = (ev[f"{src}_transit_end"] % 1) * 360.0

        if start < finish:
            axobj.axvspan(start, finish, color="grey", alpha=0.4)
        else:
            axobj.axvspan(0, finish, color="grey", alpha=0.4)
            axobj.axvspan(start, 360, color="grey", alpha=0.4)

        axobj.axvline(start, color="k", ls="--", lw=1)
        axobj.axvline(finish, color="k", ls="--", lw=1)


@_fail_quietly
def plotSens(rev, LSD, vmin=0.995, vmax=1.005):
    path = _get_rev_path("sensitivity", rev, LSD)
    sens = containers.SystemSensitivity.from_file(path)

    # Load the relevant mask
    rfm = np.zeros((sens.measured[:].shape[0], sens.measured.shape[2]), dtype=bool)
    for name in {"sens_mask"}:
        rfi_path = _get_rev_path(name, rev, LSD)
        try:
            file = containers.RFIMask.from_file(rfi_path)
        except FileNotFoundError:
            continue

        rfm |= file.mask[:]

    sp = 0

    sensrat = sens.measured[:, sp] * tools.invert_no_zero(sens.radiometer[:, sp])
    sensrat *= invert_no_zero(np.median(sensrat, axis=1))[:, np.newaxis]
    sensrat_mask = sensrat * np.where(rfm == 0, 1, np.nan)
    sensrat_mask *= np.where(_mask_flags(sens.time, LSD), np.nan, 1)

    # Select only the times that fall within the actual CSD
    sel = _select_CSD_bounds(sens.time, LSD)
    sensrat = sensrat[:, sel]
    sensrat_mask = sensrat_mask[:, sel]

    cmap = copy.copy(matplotlib.cm.viridis)
    cmap.set_bad(_BAD_VALUE_COLOR)

    # Make the master figure
    mfig = plt.figure(layout="constrained", figsize=(50, 20))
    # MAke the two sub-figures
    subfigs = mfig.subfigures(1, 2, wspace=0.1)

    for fig, sim in zip(subfigs, (sensrat, sensrat_mask)):

        mask_pcnt = 100.0 * np.isnan(sim).mean()

        axis = fig.subplots(1, 1)

        im = axis.imshow(
            sim,
            extent=(0, 360, 400, 800),
            cmap=cmap,
            aspect="auto",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        divider = make_axes_locatable(axis)
        cax = divider.append_axes("right", size="1.5%", pad=0.25)
        fig.colorbar(im, cax=cax)
        axis.set_xlabel("RA [deg]", fontsize=30)
        axis.set_ylabel("Freq [MHz]", fontsize=30)

        # Highlight relevant sources
        _highlight_sources(LSD, axis, ["sun"])

        title = _format_title(rev, LSD) + f", percent masked = {mask_pcnt:.2f}"
        axis.set_title(title, fontsize=50)
        _ = axis.set_xticks(np.arange(0, 361, 45))


def plotChisq(rev, LSD, vmin=0.9, vmax=1.4):
    path = _get_rev_path("chisq", rev, LSD)
    chisq = containers.TimeStream.from_file(path)

    vis = chisq.vis[:, 0].real

    # Load all input masks
    rfm = np.zeros(vis.shape, dtype=bool)
    for name in {"stokesi_mask", "sens_mask"}:
        rfi_path = _get_rev_path(name, rev, LSD)
        try:
            file = containers.RFIMask.from_file(rfi_path)
        except FileNotFoundError:
            continue
        rfm |= file.mask[:]

    # Load the chisq mask
    chim = np.zeros_like(rfm)
    for name in {"chisq_mask"}:
        rfi_path = _get_rev_path(name, rev, LSD)
        try:
            file = containers.RFIMask.from_file(rfi_path)
        except FileNotFoundError:
            continue
        chim |= file.mask[:]

    vis *= np.where(rfm == 0, 1, np.nan)
    vis_mask = vis * np.where(chim == 0, 1, np.nan)
    vis_mask *= np.where(_mask_flags(chisq.time, LSD), np.nan, 1)

    # Select only the times that fall within the actual CSD
    sel = _select_CSD_bounds(chisq.time, LSD)
    vis = vis[:, sel]
    vis_mask = vis_mask[:, sel]

    cmap = copy.copy(matplotlib.cm.viridis)
    cmap.set_bad(_BAD_VALUE_COLOR)

    # Make the master figure
    mfig = plt.figure(layout="constrained", figsize=(50, 20))
    # Make the two sub-figures
    subfigs = mfig.subfigures(1, 2, wspace=0.1)

    for fig, sim in zip(subfigs, (vis, vis_mask)):

        mask_pcnt = 100.0 * np.isnan(sim).mean()

        axis = fig.subplots(1, 1)

        im = axis.imshow(
            sim,
            extent=(0, 360, 400, 800),
            cmap=cmap,
            aspect="auto",
            interpolation="nearest",
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
        divider = make_axes_locatable(axis)
        cax = divider.append_axes("right", size="1.5%", pad=0.25)
        fig.colorbar(im, cax=cax)
        axis.set_xlabel("RA [deg]", fontsize=30)
        axis.set_ylabel("Freq [MHz]", fontsize=30)

        # Highlight relevant sources
        _highlight_sources(LSD, axis)

        title = _format_title(rev, LSD) + f", percent masked = {mask_pcnt:.2f}"
        axis.set_title(title, fontsize=50)
        _ = axis.set_xticks(np.arange(0, 361, 45))


@_fail_quietly
def plotVisPwr(rev, LSD, vmin=0, vmax=5e1):
    path = _get_rev_path("power", rev, LSD)
    power = containers.TimeStream.from_file(path)

    vis = power.vis[:, 0].real

    # Load the relevant RFI mask
    rfm = np.zeros(vis.shape, dtype=bool)
    for name in {"stokesi_mask"}:
        rfi_path = _get_rev_path(name, rev, LSD)
        try:
            file = containers.RFIMask.from_file(rfi_path)
        except FileNotFoundError:
            continue
        rfm |= file.mask[:]

    # Apply the initial weight mask
    vis *= np.where(power.weight[:, 0] == 0, 1, np.nan)
    # Apply the full mask
    vis_mask = vis * np.where(rfm == 0, 1, np.nan)
    vis_mask *= np.where(_mask_flags(power.time, LSD), np.nan, 1)

    # Select only the times that fall within the actual CSD
    sel = _select_CSD_bounds(power.time, LSD)
    vis = vis[:, sel]
    vis_mask = vis_mask[:, sel]

    cmap = copy.copy(matplotlib.cm.viridis)
    cmap.set_bad(_BAD_VALUE_COLOR)

    # Make the master figure
    mfig = plt.figure(layout="constrained", figsize=(50, 20))
    # MAke the two sub-figures
    subfigs = mfig.subfigures(1, 2, wspace=0.1)

    for fig, sim in zip(subfigs, (vis, vis_mask)):

        mask_pcnt = 100.0 * np.isnan(sim).mean()

        axis = fig.subplots(1, 1)

        im = axis.imshow(
            sim,
            extent=(0, 360, 400, 800),
            cmap=cmap,
            aspect="auto",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        divider = make_axes_locatable(axis)
        cax = divider.append_axes("right", size="1.5%", pad=0.25)
        fig.colorbar(im, cax=cax)
        axis.set_xlabel("RA [deg]", fontsize=30)
        axis.set_ylabel("Freq [MHz]", fontsize=30)

        # Highlight relevant sources
        sources = ["sun", "CAS_A", "CYG_A"]
        _highlight_sources(LSD, axis, sources)

        title = _format_title(rev, LSD) + f", percent masked = {mask_pcnt:.2f}"
        axis.set_title(title, fontsize=50)
        _ = axis.set_xticks(np.arange(0, 361, 45))


@_fail_quietly
def plotFactMask(rev, LSD):
    path = _get_rev_path("fact_mask", rev, LSD)
    fmask = containers.RFIMask.from_file(path)

    mask = fmask.mask[:]
    mask = np.ma.masked_where(mask == 0, mask)

    # Get the static mask that was active for this CSD
    timestamp = ephemeris.csd_to_unix(fmask.attrs.get("csd", fmask.attrs.get("lsd")))
    static_mask = np.zeros(mask.shape, mask.dtype)
    static_mask |= rfi.frequency_mask(fmask.freq, timestamp=timestamp)[:, np.newaxis]
    # Ensure that the static mask will be transparent anywhere that is not flagged
    static_mask = np.ma.masked_where(static_mask == 0, static_mask)

    # Load all the RFI masks
    for name in {"stokesi_mask", "sens_mask", "chisq_mask"}:
        rfi_path = _get_rev_path(name, rev, LSD)
        try:
            file = containers.RFIMask.from_file(rfi_path)
        except FileNotFoundError:
            continue
        try:
            rfm |= file.mask[:]
        except NameError:
            # First mask to be loaded
            rfm = file.mask[:].copy()

    # Make the master figure
    fig = plt.figure(layout="constrained", figsize=(18, 15))
    axis = fig.subplots(1, 1)

    patches = []

    # Overlay the full mask if it exists
    if "rfm" in locals():
        # Include fully flagged regions
        rfm |= _mask_flags(file.time, LSD)[np.newaxis]
        # Trim the padded time regions
        sel = _select_CSD_bounds(file.time, LSD)
        rfm = rfm[:, sel]

        cmap = matplotlib.colormaps["Reds"]
        axis.imshow(
            rfm,
            extent=(0, 360, 400, 800),
            cmap=cmap,
            aspect="auto",
            alpha=1.0,
            interpolation="nearest",
        )
        rfm_patch = mpatches.Patch(
            color=cmap(cmap.N), label=f"daily mask: {100.0 * rfm.mean():.2f}% masked"
        )
        patches.append(rfm_patch)

    # Plot the factorized mask
    cmap = matplotlib.colormaps["binary_r"]
    axis.imshow(
        mask,
        extent=(0, 360, 400, 800),
        cmap=cmap,
        aspect="auto",
        alpha=0.7,
        interpolation="nearest",
    )
    mask_patch = mpatches.Patch(
        color=cmap(0), label=f"factorized mask: {100.0 * mask.data.mean():.2f}% masked"
    )
    patches.append(mask_patch)

    # Overlay the static mask
    cmap = matplotlib.colormaps["Dark2_r"]
    axis.imshow(
        static_mask,
        extent=(0, 360, 400, 800),
        cmap=cmap,
        aspect="auto",
        alpha=1.0,
        interpolation="nearest",
    )
    static_patch = mpatches.Patch(
        color=cmap(0),
        label=f"static mask: {100.0 * static_mask.data.mean():.2f}% masked",
    )
    patches.append(static_patch)

    # Set the axes
    axis.set_xlabel("RA [deg]", fontsize=20)
    axis.set_ylabel("Freq [MHz]", fontsize=20)

    title = _format_title(rev, LSD)
    axis.set_title(title, fontsize=20)
    _ = axis.set_xticks(np.arange(0, 361, 45))

    # Add the legend
    axis.legend(
        handles=patches,
        loc=1,
        bbox_to_anchor=(1.0, -0.05),
        fancybox=True,
        ncol=2,
        shadow=True,
    )


# ========================================================================


@_fail_quietly
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

        cbar = fig.colorbar(img, ax=ax)
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
    fig = ax.figure
    trans = fig.dpi_scale_trans + matplotlib.transforms.ScaledTranslation(
        x, y, ax.transData
    )
    circle = mpatches.Circle((0, 0), radius, transform=trans, **kwargs)

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


@_fail_quietly
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

    fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(18, 15),
        gridspec_kw=dict(height_ratios=[1, 10], hspace=0.0),
    )

    fontsize = 20
    labelsize = 20

    # Plot the flagged out time ranges at the very top
    for ii, (type_, series) in enumerate(flags_by_type.items()):
        axes[0].fill_between(
            ra, ii, ii + 1, where=series, label=type_, color=f"C{ii}", alpha=0.5
        )

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
        interpolation="nearest",
        extent=(0, 360, -1, 1),
        origin="lower",
        cmap=cmap,
    )
    axes[1].set_yticks([-1, -0.5, 0, 0.5, 1])
    axes[1].yaxis.set_tick_params(labelsize=labelsize)
    axes[1].xaxis.set_tick_params(labelsize=labelsize)
    axes[1].set_ylabel("sin(ZA)", fontsize=fontsize)
    axes[1].set_xlabel("RA [degrees]", fontsize=fontsize)
    cb = plt.colorbar(
        im,
        ax=axes.ravel(),
        aspect=50,
        orientation="horizontal",
        pad=0.1,
        location="bottom",
    )

    # Put a ring around the location of the moon if it transits on this day
    if "moon_transit" in ev:
        lunar_ra = (ev["moon_transit"] % 1) * 360.0
        lunar_za = np.sin(np.radians(ev["moon_dec"] - 49.0))
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
    title = _format_title(rev, LSD) + f", {freq[0]:.2f}" + " MHz"
    axes[0].set_title(title, fontsize=fontsize)

    # Add the legend
    h1, l1 = axes[0].get_legend_handles_labels()
    fig.legend(h1, l1, loc=1)
