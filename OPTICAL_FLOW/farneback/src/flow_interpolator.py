import time

import numpy as np
from scipy.interpolate import RBFInterpolator
from tqdm import tqdm


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

    def interpolate(self, valid_coordinates: np.ndarray, valid_values: np.ndarray, height: int, width: int,
                     label: str = "") -> np.ndarray:
        prefix = f"[{label}] " if label else ""

        control_coordinates = valid_coordinates[::self.subsample_step]
        control_values = valid_values[::self.subsample_step]
        print(f"{prefix}punti validi: {len(valid_coordinates)} -> punti di controllo dopo subsampling "
              f"(1 ogni {self.subsample_step}): {len(control_coordinates)}")

        # Il fit dell'RBF risolve un sistema lineare denso di dimensione pari al numero
        # di punti di controllo: il costo cresce circa come il cubo di quel numero, quindi
        # è la fase più sensibile a quanti punti sopravvivono al subsampling.
        fit_start = time.time()
        rbf = RBFInterpolator(control_coordinates, control_values, kernel=self.kernel)
        print(f"{prefix}fit RBF completato in {time.time() - fit_start:.1f}s")

        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)
        all_coordinates = np.column_stack([xx.flatten(), yy.flatten()])

        n_chunks = (len(all_coordinates) + self.chunk_size - 1) // self.chunk_size
        predict_start = time.time()
        chunk_results = []
        for i in tqdm(range(0, len(all_coordinates), self.chunk_size), total=n_chunks,
                      desc=f"{prefix}predizione campo denso", unit="chunk", leave=False):
            chunk_results.append(rbf(all_coordinates[i:i + self.chunk_size]))
        print(f"{prefix}predizione completata in {time.time() - predict_start:.1f}s "
              f"({len(all_coordinates)} pixel totali su {n_chunks} chunk)")

        return np.vstack(chunk_results).reshape(height, width, 2)
