from pathlib import Path

import logging

logger = logging.getLogger(__name__)


BASE_PATH = Path("/project/rpp-chime/chime/chime_processed/daily")

FILE_SPEC = {
    "ringmap": ("ringmap_", (".zarr.zip", ".h5")),
    "ringmap_chisq": ("ringmap_chisq_el_template_sub_", ".h5"),
    "delayspectrum": ("delayspectrum_", ".h5"),
    "delayspectrum_hpf": ("delayspectrum_hpf_", ".h5"),
    "sensitivity": ("sensitivity_", ".h5"),
    "chisq": ("chisq_", ".h5"),
    "power": ("lowpass_power_2cyl_", ".h5"),
    "chisq_mask": ("rfi_mask_chisq_", ".h5"),
    "stokesi_mask": ("rfi_mask_stokesi_", ".h5"),
    "transient_mask": ("rfi_mask_transient_", ".h5"),
    "static_mask": ("rfi_mask_static_", ".h5"),
    "sens_mask": ("rfi_mask_sensitivity_", ".h5"),
    "freq_mask": ("rfi_mask_freq_", ".h5"),
    "fact_mask": ("rfi_mask_factorized_", ".h5"),
    "sourceflux": ("sourceflux_", "_bright.h5"),
    "sourceflux_template_subtract": ("sourceflux_template_subtract_", "_bright.h5"),
}

FREQ_BANDS = ["a", "b1", "b2", "b3", "c"]


def construct_file_path(type_: str, rev: int, lsd: int) -> Path:
    if type_ not in FILE_SPEC:
        raise ValueError(f"Unknown file type {type_}.")

    prefix, suffix = FILE_SPEC[type_]

    if not isinstance(suffix, list | tuple):
        suffix = [suffix]

    for sfx in suffix:
        candidate_path = (
            BASE_PATH / f"rev_{rev:02d}" / f"{lsd:d}" / f"{prefix}lsd_{lsd:d}{sfx}"
        )

        if candidate_path.exists():
            return candidate_path

    raise FileNotFoundError(
        f"No file found for type {type_}, rev {rev}, lsd {lsd}, "
        f"with candidate prefix/suffix(es) {prefix}/{suffix}."
    )


def construct_file_path_for_bands(type_: str, rev: int, lsd: int) -> list[Path]:
    # Get the standard path for this type
    if type_ not in FILE_SPEC:
        raise ValueError(f"Unknown file type {type_}.")

    prefix, suffix = FILE_SPEC[type_]

    if isinstance(prefix, list | tuple):
        if len(prefix) > 1:
            raise ValueError(
                f"Cannot construct band paths for types with multiple options. Got {prefix}."
            )
        else:
            prefix = prefix[0]

    if isinstance(suffix, list | tuple):
        if len(suffix) > 1:
            raise ValueError(
                f"Cannot construct band paths for types with multiple options. Got {suffix}."
            )
        else:
            suffix = suffix[0]

    fpath = BASE_PATH / f"rev_{rev:02d}" / f"{lsd:d}" / f"{prefix}lsd_{lsd:d}{suffix}"

    new_paths = []

    for band in FREQ_BANDS:
        new_paths.append(fpath.with_stem(fpath.stem + f"_band{band}"))

    return new_paths
