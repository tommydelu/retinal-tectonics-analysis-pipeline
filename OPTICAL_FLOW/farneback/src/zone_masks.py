import numpy as np


class RetinalZoneMasks:
    """
    Maschere geometriche centrate sulla fovea, usate per aggregare le metriche di optical
    flow per regione anatomica: intera immagine, ROI totale (anello interno + esterno),
    anello interno, anello esterno, e i quattro quadranti di ciascun anello.
    """

    def __init__(self, height, width, fovea_center, inner_radius, outer_radius):
        cx, cy = fovea_center
        y_coords, x_coords = np.indices((height, width))
        dx = x_coords - cx
        dy = y_coords - cy
        distance = np.sqrt(dx**2 + dy**2)
        angle = np.rad2deg(np.arctan2(-dy, dx))

        self.entire_image = np.ones((height, width), dtype=bool)
        self.inner = distance <= inner_radius
        self.outer = (distance > inner_radius) & (distance <= outer_radius)
        self.total_roi = self.inner | self.outer

        # Quadranti geometrici (right/top/left/bottom secondo l'angolo standard, 0° = destra)
        mask_right = (angle >= -45) & (angle < 45)
        mask_top = (angle >= 45) & (angle < 135)
        mask_left = (angle >= 135) | (angle < -135)
        mask_bottom = (angle >= -135) & (angle < -45)

        # NB: l'etichetta numerica ("45", "135", "225", "315") non corrisponde all'angolo
        # geometrico del quadrante, ma alla convenzione già usata nei CSV esistenti del
        # progetto (45=bottom, 135=left, 225=top, 315=right) — mappatura preservata invariata.
        self._quadrants = {
            "45": mask_bottom,
            "135": mask_left,
            "225": mask_top,
            "315": mask_right,
        }

    def named_regions(self) -> dict[str, np.ndarray]:
        """
        Ritorna {suffisso_colonna: maschera} per tutte le regioni usate nei CSV di
        output: entire image, total ROI, inner avg, outer avg, e gli 8 quadranti
        (4 dell'anello interno + 4 dell'anello esterno).
        """
        regions = {
            "entire image": self.entire_image,
            "total ROI": self.total_roi,
            "inner avg": self.inner,
            "outer avg": self.outer,
        }
        for angle_label, quadrant_mask in self._quadrants.items():
            regions[f"{angle_label} inner"] = self.inner & quadrant_mask
            regions[f"{angle_label} outer"] = self.outer & quadrant_mask
        return regions
