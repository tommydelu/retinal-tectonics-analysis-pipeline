import numpy as np

from common.subject_data import safe_mean
from OPTICAL_FLOW.farneback.src.zone_masks import RetinalZoneMasks


def compute_zone_metrics(u: np.ndarray, v: np.ndarray, zones: RetinalZoneMasks,
                          pixel_length_x: float, pixel_length_y: float, valid_mask: np.ndarray = None) -> dict:
    """
    Calcola la media di X (u), Y (v) e magnitudine dell'optical flow per ciascuna
    regione anatomica di `zones`, convertita in micrometri.

    u e v vengono convertiti separatamente con pixel_length_x e pixel_length_y (possono
    differire per risoluzioni non quadrate): la magnitudine viene poi ricavata dai
    componenti già convertiti (sqrt(u_um^2 + v_um^2)), non da una magnitudine in pixel
    scalata con un unico fattore, perché coi due assi anisotropi le due cose non
    coincidono.

    Se `valid_mask` è None, la media è calcolata su tutta la regione geometrica
    (modalità "interpolated": il campo è denso, definito ovunque). Se `valid_mask`
    è fornita, viene intersecata con ciascuna regione prima di mediare (modalità
    "masked": si media solo sui pixel di vaso effettivamente validi).

    Per ogni regione viene riportato anche "n_valid {suffix}", il numero di pixel su cui
    è stata calcolata la media: se è 0, X/Y/magnitude di quella regione sono NaN (nessun
    pixel di vaso valido in quella zona, non una misura di movimento nullo), e valori
    piccoli segnalano una media comunque poco affidabile.
    """

    u_um = u * pixel_length_x
    v_um = v * pixel_length_y
    magnitude_um = np.sqrt(u_um**2 + v_um**2)

    metrics = {}
    for suffix, region_mask in zones.named_regions().items():
        mask = region_mask if valid_mask is None else (region_mask & valid_mask)

        metrics[f"X {suffix}"] = safe_mean(u_um, mask)
        metrics[f"Y {suffix}"] = safe_mean(v_um, mask)
        metrics[f"magnitude {suffix}"] = safe_mean(magnitude_um, mask)
        metrics[f"n_valid {suffix}"] = int(np.count_nonzero(mask))

    return metrics
