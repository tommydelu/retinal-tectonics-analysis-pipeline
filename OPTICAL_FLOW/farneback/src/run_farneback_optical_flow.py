"""
Calcolo dell'optical flow di Farneback tra immagini PRE/POST, su dataset 1 o dataset 2
(quest'ultimo limitato al confronto baseline vs 12 mesi post-operatorio), in due
modalità:

- "interpolated": il campo di flow viene calcolato solo sui pixel di vaso affidabili
  e poi interpolato (RBF, thin plate spline) su tutta l'immagine, per ottenere un
  campo denso e più pulito dal rumore del background.
- "masked": nessuna interpolazione — le metriche vengono calcolate direttamente sul
  campo di flow grezzo di Farneback, ristretto ai soli pixel di vaso affidabili.

La maschera dei vasi può venire da due sorgenti (--mask-source):

- "gt" (default): label manuali (ground-truth), su TUTTI i soggetti del dataset scelto.
- "auto": output della segmentazione automatica. Disponibile solo per il dataset 1, e
  solo sui 10 soggetti con Dice Score migliore (BEST_SUBJECTS in subjects.py), perché
  è l'unico sottoinsieme per cui la segmentazione automatica è stata generata.

Esempio:
    python -m OPTICAL_FLOW.farneback.src.run_farneback_optical_flow --dataset 1 --mode interpolated --mask-source gt
    python -m OPTICAL_FLOW.farneback.src.run_farneback_optical_flow --dataset 1 --mode masked --mask-source auto --save-figures
    python -m OPTICAL_FLOW.farneback.src.run_farneback_optical_flow --dataset 2 --mode masked --save-figures
"""

import os
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from common.paths import PROJECT_ROOT
from OPTICAL_FLOW.utils.config import MAX_ALLOWED_MOVEMENT, PIXEL_LENGTH, INNER_RADIUS, OUTER_RADIUS
from OPTICAL_FLOW.farneback.src.subjects import Dataset1Subjects, Dataset2Subjects
from OPTICAL_FLOW.farneback.src.flow_field import FarnebackFlowField
from OPTICAL_FLOW.farneback.src.flow_interpolator import RBFFlowInterpolator
from OPTICAL_FLOW.farneback.src.zone_masks import RetinalZoneMasks
from OPTICAL_FLOW.farneback.src.flow_metrics import compute_zone_metrics
from OPTICAL_FLOW.farneback.src.figures import save_quiver_figure


IMAGE_TYPE = "ir_raw"  # sfondo su cui disegnare i vettori: IR pre-op grezza, senza CLAHE


def _output_paths(dataset: int, mask_source: str):
    dataset_base_path = os.path.join(PROJECT_ROOT, 'OPTICAL_FLOW', 'farneback', 'results', f'ds{dataset}')
    results_data_path = os.path.join(dataset_base_path, 'data')
    figures_path = os.path.join(dataset_base_path, 'figures', mask_source, IMAGE_TYPE)
    npy_cache_path = os.path.join(results_data_path, 'interpolated_cache', mask_source)

    os.makedirs(results_data_path, exist_ok=True)
    os.makedirs(npy_cache_path, exist_ok=True)

    return results_data_path, figures_path, npy_cache_path


def _dense_field_for_subject(flow: FarnebackFlowField, subject_id: str, comparison_label: str,
                              npy_cache_path: str, interpolator: RBFFlowInterpolator) -> np.ndarray:
    """Campo denso interpolato, riusato da cache su disco se già calcolato in una run precedente."""
    cache_file = os.path.join(npy_cache_path, f"{subject_id}_{comparison_label}_interpolated_flow.npy")

    if os.path.exists(cache_file):
        print(f"[{subject_id} {comparison_label}] campo denso già in cache, salto l'interpolazione")
        return np.load(cache_file)

    height, width = flow.magnitude.shape
    valid_coordinates, valid_values = flow.valid_points()
    dense_field = interpolator.interpolate(valid_coordinates, valid_values, height, width,
                                            label=f"{subject_id} {comparison_label}")
    dense_field = FarnebackFlowField.clip_to_max_movement(dense_field, MAX_ALLOWED_MOVEMENT)

    np.save(cache_file, dense_field)
    return dense_field


def run(dataset: int, mode: str, mask_source: str = "gt", save_figures: bool = False) -> pd.DataFrame:
    if dataset == 2 and mask_source == "auto":
        raise ValueError(
            "Il dataset 2 non ha maschere di segmentazione automatica: usa --mask-source gt."
        )

    subjects_source = Dataset1Subjects(mask_source=mask_source) if dataset == 1 else Dataset2Subjects()
    results_data_path, figures_path, npy_cache_path = _output_paths(dataset, mask_source)
    interpolator = RBFFlowInterpolator()

    all_results = []

    for subject in tqdm(subjects_source, desc=f"Dataset {dataset} ({mask_source}, {mode})"):
        flow = FarnebackFlowField(subject.img_pre, subject.img_post,
                                   subject.vessel_mask_pre, subject.vessel_mask_post,
                                   MAX_ALLOWED_MOVEMENT)

        height, width = flow.magnitude.shape
        zones = RetinalZoneMasks(height, width, subject.fovea_center, INNER_RADIUS, OUTER_RADIUS)

        if mode == "interpolated":
            dense_field = _dense_field_for_subject(flow, subject.id, subject.comparison_label, npy_cache_path, interpolator)
            u, v = dense_field[:, :, 0], dense_field[:, :, 1]
            magnitude = np.sqrt(u**2 + v**2)
            metrics = compute_zone_metrics(u, v, magnitude, zones, PIXEL_LENGTH, valid_mask=None)
        else:  # masked
            u, v = flow.raw_field[:, :, 0], flow.raw_field[:, :, 1]
            metrics = compute_zone_metrics(u, v, flow.magnitude, zones, PIXEL_LENGTH, valid_mask=flow.valid_mask)

        all_results.append({
            "dataset": dataset,
            "subject": subject.id,
            "comparison": subject.comparison_label,
            **metrics,
        })

        if save_figures:
            fig_path = os.path.join(figures_path, f"{subject.id}_{mode}.png")
            save_quiver_figure(subject, u, v, zones, INNER_RADIUS, OUTER_RADIUS, fig_path)

    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(results_data_path, f"dataset{dataset}_{mask_source}_{mode}_metrics.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nMetriche salvate in: {csv_path}")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=int, choices=[1, 2], required=True, help="Dataset su cui calcolare l'optical flow")
    parser.add_argument("--mode", choices=["interpolated", "masked"], required=True,
                         help="'interpolated' = interpolazione RBF sui vasi; 'masked' = solo sui pixel dei vasi, senza interpolazione")
    parser.add_argument("--mask-source", choices=["gt", "auto"], default="gt",
                         help="'gt' = maschere ground-truth, su tutti i soggetti (default); "
                              "'auto' = maschere di segmentazione automatica, solo dataset 1, limitate ai 10 soggetti con Dice Score migliore")
    parser.add_argument("--save-figures", action="store_true", help="Salva anche le figure con il campo vettoriale sovrapposto")
    args = parser.parse_args()

    run(dataset=args.dataset, mode=args.mode, mask_source=args.mask_source, save_figures=args.save_figures)
