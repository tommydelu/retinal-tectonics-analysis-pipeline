import cv2 as cv
import numpy as np


class FarnebackFlowField:

    def __init__(self, img_pre: np.ndarray, img_post: np.ndarray, vessel_mask_pre: np.ndarray,
                 vessel_mask_post: np.ndarray, winsize: int):

        self.raw_field = cv.calcOpticalFlowFarneback(
            prev=img_pre, next=img_post, flow=None,
            pyr_scale=0.5, levels=5, winsize=winsize, iterations=10,
            poly_n=7, poly_sigma=1.5, flags=0,
        )
        self.magnitude = np.sqrt(self.raw_field[:, :, 0]**2 + self.raw_field[:, :, 1]**2)
        self.vessel_mask = vessel_mask_pre.astype(bool) & vessel_mask_post.astype(bool)

    def valid_mask_for_threshold(self, min_allowed_um: float, max_allowed_um: float, 
                                 pixel_length: float) -> np.ndarray:
        """Filtra i pixel in base alle soglie fisiche in MICROMETRI."""
        # Convertiamo la magnitudine (che è in pixel) in micrometri
        magnitude_um = self.magnitude * pixel_length
        
        magnitude_mask = (magnitude_um >= min_allowed_um) & (magnitude_um < max_allowed_um)
        return self.vessel_mask & magnitude_mask

    def valid_points(self, valid_mask: np.ndarray):
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
    def clip_to_max_movement(field: np.ndarray, max_allowed_um: float, pixel_length: float) -> np.ndarray:
        """Taglia gli overshoot dell'interpolazione usando la soglia in MICROMETRI."""
        field = field.copy()
        u, v = field[:, :, 0], field[:, :, 1]
        magnitude = np.sqrt(u**2 + v**2)
        
        max_pixel_movement = max_allowed_um / pixel_length

        overshoot = magnitude > max_pixel_movement
        scale_factor = max_pixel_movement / magnitude[overshoot]
        field[overshoot, 0] = u[overshoot] * scale_factor
        field[overshoot, 1] = v[overshoot] * scale_factor
        return field