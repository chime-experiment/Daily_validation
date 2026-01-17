from datetime import datetime

from ch_ephem.observers import chime as chime_obs

from chimedb import core
import chimedb.dataflag as df

import logging

logger = logging.getLogger(__name__)


DATA_FLAGS = [
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


def fail_quietly(func):
    """Just log any exceptions."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as err:
            logger.debug(f"Function {func.__name__} failed with error:\n{err}")
            return None

    return wrapper


def get_data_flags(LSD):
    core.connect()

    ut_start = chime_obs.lsd_to_unix(LSD)
    ut_end = chime_obs.lsd_to_unix(LSD + 1)

    flags = (
        df.DataFlag.select()
        .where(df.DataFlag.start_time < ut_end, df.DataFlag.finish_time > ut_start)
        .join(df.DataFlagType)
        .where(df.DataFlagType.name << DATA_FLAGS)
    )

    flag_time_spans = [(f.type.name, f.start_time, f.finish_time) for f in flags]

    return flag_time_spans


def query_calibrator(LSD):
    """Query the (likely) active calibrator for a CSD."""
    import chimedb.dataset as ds

    core.connect()

    gain_type = ds.DatasetStateType.select().where(ds.DatasetStateType.name == "gains")

    query = (
        ds.DatasetState.select(ds.DatasetState.id)
        .where(
            ds.DatasetState.type == gain_type,
            ds.DatasetState.time <= datetime.fromtimestamp(chime_obs.lsd_to_unix(LSD)),
        )
        .order_by(
            ds.DatasetState.time.desc(),
        )
    )

    # Get the closest result, which should be the applied gains for this CSD
    if not query:
        return "Unknown"

    # Work backwards to get the most recent calibrator
    for entry in query:
        update_id = ds.DatasetState.from_id(entry.id).data["data"]["update_id"]

        components = update_id.split("_")

        # `components` should have at least length 3 if this is valid
        if len(components) < 3:
            continue

        calibrator = components[2].upper()
        return calibrator[:-1] + "_" + calibrator[-1]

    return "Unknown"
