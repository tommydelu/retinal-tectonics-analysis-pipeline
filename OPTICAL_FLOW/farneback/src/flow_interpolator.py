import numpy as np
from scipy.interpolate import RBFInterpolator


class RBFFlowInterpolator:
    """
    Interpola un campo vettoriale sparso (calcolato solo sui pixel affidabili, cioè
    quelli di vaso con spostamento plausibile) su tutta l'immagine tramite una Radial
    Basis Function (thin plate spline di default), per ottenere un campo di optical
    flow denso e più pulito dal rumore del background.
    """

    def __init__(self, kernel: str = "thin_plate_spline", subsample_step: int = 30, chunk_size: int = 30000):
        self.kernel = kernel
        self.subsample_step = subsample_step
        self.chunk_size = chunk_size

    def interpolate(self, valid_coordinates: np.ndarray, valid_values: np.ndarray, height: int, width: int) -> np.ndarray:
        rbf = RBFInterpolator(
            valid_coordinates[::self.subsample_step],
            valid_values[::self.subsample_step],
            kernel=self.kernel,
        )

        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)
        all_coordinates = np.column_stack([xx.flatten(), yy.flatten()])

        chunk_results = []
        for i in range(0, len(all_coordinates), self.chunk_size):
            chunk_results.append(rbf(all_coordinates[i:i + self.chunk_size]))

        return np.vstack(chunk_results).reshape(height, width, 2)
