import cv2 as cv
import numpy as np


class FarnebackFlowField:
    """
    Campo di optical flow denso calcolato con l'algoritmo di Farneback tra due immagini
    (PRE e POST), insieme alla maschera dei pixel sui vasi in entrambe le fasi.

    Il calcolo di Farneback (costoso) viene fatto una sola volta per soggetto; la
    maschera dei pixel affidabili in base alla soglia di movimento min/max si ottiene
    poi con `valid_mask_for_threshold`, richiamabile più volte per soglie diverse senza
    ripetere il calcolo del campo.
    """

    def __init__(self, img_pre: np.ndarray, img_post: np.ndarray, vessel_mask_pre: np.ndarray,
                 vessel_mask_post: np.ndarray, winsize: int):

        self.raw_field = cv.calcOpticalFlowFarneback(
            prev=img_pre, next=img_post, flow=None,
            pyr_scale=0.5, levels=5, winsize=winsize, iterations=10,
            poly_n=7, poly_sigma=1.5, flags=0,
        )

        self.magnitude = np.sqrt(self.raw_field[:, :, 0]**2 + self.raw_field[:, :, 1]**2)
        self.vessel_mask = vessel_mask_pre.astype(bool) & vessel_mask_post.astype(bool)

    def valid_mask_for_threshold(self, min_allowed_movement: float, max_allowed_movement: float) -> np.ndarray:
        """Maschera dei pixel di vaso il cui spostamento è nel range [min, max] plausibile."""
        magnitude_mask = (self.magnitude >= min_allowed_movement) & (self.magnitude < max_allowed_movement)
        return self.vessel_mask & magnitude_mask

    def valid_points(self, valid_mask: np.ndarray):
        """Coordinate (x, y) e valori (u, v) del campo, solo ai pixel validi — input dell'interpolatore RBF."""
        height, width = self.magnitude.shape
        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)

        xx_valid = xx[valid_mask]
        yy_valid = yy[valid_mask]

        valid_coordinates = np.column_stack([xx_valid, yy_valid])
        valid_values = self.raw_field[valid_mask]

        return valid_coordinates, valid_values

    @staticmethod
    def clip_to_max_movement(field: np.ndarray, max_allowed_movement: float) -> np.ndarray:
        """Rescala i vettori il cui modulo supera la soglia massima plausibile (tipico artefatto dell'interpolazione)."""
        field = field.copy()
        u, v = field[:, :, 0], field[:, :, 1]
        magnitude = np.sqrt(u**2 + v**2)

        overshoot = magnitude > max_allowed_movement
        scale_factor = max_allowed_movement / magnitude[overshoot]
        field[overshoot, 0] = u[overshoot] * scale_factor
        field[overshoot, 1] = v[overshoot] * scale_factor

        return field
