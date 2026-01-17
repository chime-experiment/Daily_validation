from datetime import datetime, timezone

from ch_util import cal_utils
from ch_ephem.observers import chime as chime_obs
from ch_ephem.sources import source_dictionary
from caput.astro import skyfield, time as ctime

eph = skyfield.skyfield_wrapper.ephemeris

SOURCES = ["sun", "moon", "CAS_A", "CYG_A", "TAU_A", "VIR_A", "B0329+54"]


def select_CSD_bounds(times, LSD, obs=chime_obs):
    """Return indices representing the times found within the LSD"""
    ra = obs.unix_to_lsd(times) - LSD

    return (ra >= 0.0) & (ra < 1.0)


def get_extent(times, LSD, obs=chime_obs):
    """Convert times to fractional CSD bounds."""
    ra = 360 * (obs.unix_to_lsd(times) - LSD)

    return ra[0], ra[-1]


def get_csd(day: int | str | None = None, num_days: int = 0, lag: int = 0) -> int:
    """Get a csd from an integer or a string with format yyyy/mm/dd.

    If None, return the current CSD.
    """
    if day is None:
        return int(chime_obs.get_current_lsd() - num_days - lag)

    if isinstance(day, str):
        day = datetime.strptime(day, "%Y/%m/%d").timestamp()
        return int(chime_obs.unix_to_lsd(day))

    return int(day)


def csd_to_utc(csd: int | str, include_time: bool = False) -> str:
    """Convert a CSD to a formatted UTC day."""
    date = chime_obs.lsd_to_unix(int(csd))

    if include_time:
        fmt = "%Y/%m/%d %H/%M/%S"
    else:
        fmt = "%Y/%m/%d"

    return datetime.fromtimestamp(date, tz=timezone.utc).strftime(fmt)


def events(observer=chime_obs, lsd=None):
    if lsd is None:
        return {}, []

    # Start and end times of the CSD
    st = observer.lsd_to_unix(lsd)
    et = observer.lsd_to_unix(lsd + 1)

    e = {}
    return_sources = []

    u2l = observer.unix_to_lsd

    sources = [src for src in SOURCES if src not in {"sun", "moon"}]

    bodies = {src: source_dictionary[src] for src in sources}
    bodies["moon"] = eph["moon"]

    # Sun is handled differently because we care about rise/set
    # rather than just transit

    tt = observer.transit_times(eph["sun"], st, et)

    if tt.size > 0:
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

        if tt.size > 0:
            tt = tt[0]
        else:
            continue

        sf_time = ctime.unix_to_skyfield_time(tt)
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
            window_sec = window_deg * 240.0 * ctime.SIDEREAL_S

            # Enter the transit timings in the output dict
            e[f"{name}_transit"] = u2l(tt)
            e[f"{name}_transit_start"] = u2l(tt - window_sec)
            e[f"{name}_transit_end"] = u2l(tt + window_sec)
            # Record that there is transit information for this body
            return_sources.append(name)

    return e, ["sun"] + return_sources
