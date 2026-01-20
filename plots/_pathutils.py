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
    "transient_mask": ("rfi_transient_mask_", ".h5"),
    "static_mask": ("rfi_static_mask_", "h5"),
    "sens_mask": ("rfi_mask_sensitivity_", ".h5"),
    "freq_mask": ("rfi_mask_freq_", ".h5"),
    "fact_mask": ("rfi_mask_factorized_", ".h5"),
    "sourceflux": ("sourceflux_", "_bright.h5"),
    "sourceflux_template_subtract": ("sourceflux_template_subtract_", "_bright.h5"),
}

FREQ_BANDS = ["a", "b1", "b2", "b3", "c"]


def construct_file_path(
    type_: str, rev: int, lsd: int, require_exists: bool = True
) -> Path:
    if type_ not in FILE_SPEC:
        raise ValueError(f"Unknown file type {type_}.")

    prefix, suffix = FILE_SPEC[type_]

    if not isinstance(suffix, list | tuple):
        suffix = [suffix]

    for sfx in suffix:
        candidate_path = (
            BASE_PATH / f"rev_{rev:02d}" / f"{lsd:d}" / f"{prefix}lsd_{lsd:d}{sfx}"
        )

    if require_exists and not candidate_path.exists():
        raise FileNotFoundError(
            f"No file found for type {type_}, rev {rev}, lsd {lsd}, "
            f"with candidate prefix/suffix(es) {prefix}/{suffix}."
        )

    return candidate_path


def construct_file_path_for_bands(type_: str, rev: int, lsd: int) -> list[Path]:
    # Get the standard path for this type
    fpath = construct_file_path(type_, rev, lsd, require_exists=False)

    new_paths = []

    for band in FREQ_BANDS:
        new_paths.append(fpath.with_stem(fpath.stem + f"_band{band}"))

    return new_paths
