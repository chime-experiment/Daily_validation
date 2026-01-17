from pathlib import Path

import logging

logger = logging.getLogger(__name__)


BASE_PATH = Path("/project/rpp-chime/chime/chime_processed/daily")

FILE_SPEC = {
    "ringmap": ("ringmap_", (".zarr.zip", ".h5")),
    "delayspectrum": ("delayspectrum_", ".h5"),
    "delayspectrum_hpf": ("delayspectrum_hpf_", ".h5"),
    "sensitivity": ("sensitivity_", ".h5"),
    "chisq": ("chisq_", ".h5"),
    "power": ("lowpass_power_2cyl_", ".h5"),
    "chisq_mask": ("rfi_mask_chisq_", ".h5"),
    "stokesi_mask": ("rfi_mask_stokesi_", ".h5"),
    "sens_mask": ("rfi_mask_sensitivity_", ".h5"),
    "freq_mask": ("rfi_mask_freq_", ".h5"),
    "fact_mask": ("rfi_mask_factorized_", ".h5"),
    "sourceflux": ("sourceflux_", "_bright.h5"),
}


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
