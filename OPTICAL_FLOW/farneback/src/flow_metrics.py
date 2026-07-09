import numpy as np

from common.subject_data import safe_mean
from OPTICAL_FLOW.farneback.src.zone_masks import RetinalZoneMasks


def compute_zone_metrics(u: np.ndarray, v: np.ndarray, magnitude: np.ndarray, zones: RetinalZoneMasks,
                          pixel_length: float, valid_mask: np.ndarray = None) -> dict:
    """
    Calcola la media di X (u), Y (v) e magnitudine dell'optical flow per ciascuna
    regione anatomica di `zones`, convertita in micrometri (pixel_length).

    Se `valid_mask` è None, la media è calcolata su tutta la regione geometrica
    (modalità "interpolated": il campo è denso, definito ovunque). Se `valid_mask`
    è fornita, viene intersecata con ciascuna regione prima di mediare (modalità
    "masked": si media solo sui pixel di vaso effettivamente validi).
    """

    metrics = {}
    for suffix, region_mask in zones.named_regions().items():
        mask = region_mask if valid_mask is None else (region_mask & valid_mask)

        metrics[f"X {suffix}"] = safe_mean(u, mask) * pixel_length
        metrics[f"Y {suffix}"] = safe_mean(v, mask) * pixel_length
        metrics[f"magnitude {suffix}"] = safe_mean(magnitude, mask) * pixel_length

    return metrics
