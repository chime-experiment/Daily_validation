import numpy as np

import matplotlib
import matplotlib.patches as mpatches

from . import _ephemutils, _util

BAD_VALUE_COLOR = "#1a1a1a"


def format_title(rev, LSD):
    """Return a title string for plots."""
    return f"rev_{int(rev):02d}, CSD {int(LSD):04d}"


def mask_baselines(baseline_vec, single_mask=False):
    """Mask long baselines in a delay spectrum."""

    bl_mask = np.zeros((4, baseline_vec.shape[0]), dtype=bool)
    bl_mask[0] = baseline_vec[:, 0] < 10
    bl_mask[1] = (baseline_vec[:, 0] > 10) & (baseline_vec[:, 0] < 30)
    bl_mask[2] = (baseline_vec[:, 0] > 30) & (baseline_vec[:, 0] < 50)
    bl_mask[3] = baseline_vec[:, 0] > 50

    if single_mask:
        bl_mask = np.any(bl_mask, axis=0)

    return bl_mask


def hide_axis_ticks(ax):
    """Hide axis ticks and frame without removing axis."""

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(left=False, bottom=False)


def mask_flags(times, LSD):
    flag_mask = np.zeros_like(times, dtype=bool)

    for type_, ca, cb in _util.get_data_flags(LSD):
        flag_mask[(times > ca) & (times < cb)] = True

    return flag_mask


def highlight_sources(LSD, axobj, sources=_ephemutils.SOURCES):
    """Add shaded regions over source objects."""
    ev, srcs = _ephemutils.events(lsd=LSD)

    for src in sources:
        if src not in srcs:
            # No data was available for this source
            continue
        if src == "sun":
            start = (ev["sun_rise"] % 1) * 360.0 if "sun_rise" in ev else 0
            finish = (ev["sun_set"] % 1) * 360.0 if "sun_set" in ev else 360
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


def draw_circle(ax, x, y, radius, **kwargs):
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


def infill_gaps(data, x, y, xspan=None, yspan=None):
    """Infill a dataset with missing samples with NaNs."""

    def _interp(xi, span):
        if span is None:
            span = (xi[0], xi[-1])

        dx = np.gradient(xi, axis=-1)
        dxm = np.median(dx)

        # Make a constant grid
        xp = np.arange(span[0], span[1] + dxm, dxm)
        # Map the existing data values to their indices
        # in the new grid
        map_ = abs(xp[:, np.newaxis] - xi[np.newaxis]).argmin(axis=0)
        # Replace with the actual values
        xp[map_] = xi

        return xp, map_

    xp, xmap = _interp(x, xspan)
    yp, ymap = _interp(y, yspan)

    sel_ = len(yp) * xmap[np.newaxis] + ymap[:, np.newaxis]
    sel_ = sel_.ravel(order="C")

    newdata = np.full((len(xp), len(yp)), np.nan, data.dtype)
    newdata.ravel(order="C")[sel_] = data.ravel(order="C")

    return newdata.T, xp, yp
