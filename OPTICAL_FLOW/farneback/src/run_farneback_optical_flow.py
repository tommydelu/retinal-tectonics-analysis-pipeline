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

Il campo di Farneback viene calcolato una sola volta per soggetto e poi rifiltrato per
ciascuna coppia di soglie (min, max) di MOVEMENT_THRESHOLDS (config.py) o passata da
riga di comando con --thresholds: min = spostamento minimo plausibile (sotto è rumore
statico), max = spostamento massimo plausibile (sopra è outlier). Ogni coppia di soglie
produce un CSV di metriche separato.

La pixel_length (um/pixel) è determinata per-immagine dalla risoluzione: per il dataset 2,
dove le risoluzioni variano da soggetto a soggetto e talvolta non sono quadrate, viene
cercata in PIXEL_LENGTH_MAPPING (config.py) con valori separati per asse x e y; se la
risoluzione non è mappata (dataset 1, sempre alla stessa risoluzione) si usa PIXEL_LENGTH.
Da questa vengono ricavati inner/outer radius, anch'essi separati per asse x e y.

Esempio:
    python -m OPTICAL_FLOW.farneback.src.run_farneback_optical_flow --dataset 1 --mode interpolated --mask-source gt
    python -m OPTICAL_FLOW.farneback.src.run_farneback_optical_flow --dataset 1 --mode masked --mask-source auto --save-figures
    python -m OPTICAL_FLOW.farneback.src.run_farneback_optical_flow --dataset 2 --mode masked --save-figures
    python -m OPTICAL_FLOW.farneback.src.run_farneback_optical_flow --dataset 1 --mode masked --thresholds 1-10 2-15 3-20
"""

import os
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from common.paths import PROJECT_ROOT
from OPTICAL_FLOW.utils.config import MOVEMENT_THRESHOLDS, get_pixel_length, compute_radii
from OPTICAL_FLOW.farneback.src.subjects import Dataset1Subjects, Dataset2Subjects
from OPTICAL_FLOW.farneback.src.flow_field import FarnebackFlowField
from OPTICAL_FLOW.farneback.src.flow_interpolator import RBFFlowInterpolator
from OPTICAL_FLOW.farneback.src.zone_masks import RetinalZoneMasks
from OPTICAL_FLOW.farneback.src.flow_metrics import compute_zone_metrics
from OPTICAL_FLOW.farneback.src.figures import save_quiver_figure


IMAGE_TYPE = "ir_raw"  # sfondo su cui disegnare i vettori: IR pre-op grezza, senza CLAHE


def _threshold_tag(min_move: float, max_move: float) -> str:
    return f"min{min_move:g}_max{max_move:g}"


def _output_paths(dataset: int, mask_source: str):
    dataset_base_path = os.path.join(PROJECT_ROOT, 'OPTICAL_FLOW', 'farneback', 'results', f'ds{dataset}')
    results_data_path = os.path.join(dataset_base_path, 'data')
    figures_path = os.path.join(dataset_base_path, 'figures', mask_source, IMAGE_TYPE)
    npy_cache_path = os.path.join(results_data_path, 'interpolated_cache', mask_source)

    os.makedirs(results_data_path, exist_ok=True)
    os.makedirs(npy_cache_path, exist_ok=True)

    return results_data_path, figures_path, npy_cache_path


def _dense_field_for_subject(flow: FarnebackFlowField, valid_mask: np.ndarray, subject_id: str, comparison_label: str,
                              min_move: float, max_move: float, npy_cache_path: str,
                              interpolator: RBFFlowInterpolator) -> np.ndarray:
    """Campo denso interpolato, riusato da cache su disco se già calcolato in una run precedente."""
    tag = _threshold_tag(min_move, max_move)
    cache_file = os.path.join(npy_cache_path, f"{subject_id}_{comparison_label}_{tag}_interpolated_flow.npy")

    if os.path.exists(cache_file):
        print(f"[{subject_id} {comparison_label} {tag}] campo denso già in cache, salto l'interpolazione")
        return np.load(cache_file)

    height, width = flow.magnitude.shape
    valid_coordinates, valid_values = flow.valid_points(valid_mask)
    dense_field = interpolator.interpolate(valid_coordinates, valid_values, height, width,
                                            label=f"{subject_id} {comparison_label} {tag}")
    dense_field = FarnebackFlowField.clip_to_max_movement(dense_field, max_move)

    np.save(cache_file, dense_field)
    return dense_field


def run(dataset: int, mode: str, mask_source: str = "gt", save_figures: bool = False,
        thresholds: list[tuple[float, float]] = None) -> dict[tuple[float, float], pd.DataFrame]:
    if dataset == 2 and mask_source == "auto":
        raise ValueError(
            "Il dataset 2 non ha maschere di segmentazione automatica: usa --mask-source gt."
        )

    thresholds = thresholds or MOVEMENT_THRESHOLDS

    subjects_source = Dataset1Subjects(mask_source=mask_source) if dataset == 1 else Dataset2Subjects()
    results_data_path, figures_path, npy_cache_path = _output_paths(dataset, mask_source)
    interpolator = RBFFlowInterpolator()

    all_results = {threshold: [] for threshold in thresholds}

    for subject in tqdm(subjects_source, desc=f"Dataset {dataset} ({mask_source}, {mode})"):
        flow = FarnebackFlowField(subject.img_pre, subject.img_post,
                                   subject.vessel_mask_pre, subject.vessel_mask_post)

        height, width = flow.magnitude.shape
        pixel_length_x, pixel_length_y = get_pixel_length(width, height)
        inner_radius_x, inner_radius_y, outer_radius_x, outer_radius_y = compute_radii(pixel_length_x, pixel_length_y)
        zones = RetinalZoneMasks(height, width, subject.fovea_center,
                                  inner_radius_x, inner_radius_y, outer_radius_x, outer_radius_y)

        for min_move, max_move in thresholds:
            tag = _threshold_tag(min_move, max_move)
            valid_mask = flow.valid_mask_for_threshold(min_move, max_move)

            if mode == "interpolated":
                dense_field = _dense_field_for_subject(flow, valid_mask, subject.id, subject.comparison_label,
                                                         min_move, max_move, npy_cache_path, interpolator)
                u, v = dense_field[:, :, 0], dense_field[:, :, 1]
                metrics = compute_zone_metrics(u, v, zones, pixel_length_x, pixel_length_y, valid_mask=None)
            else:  # masked
                u, v = flow.raw_field[:, :, 0], flow.raw_field[:, :, 1]
                metrics = compute_zone_metrics(u, v, zones, pixel_length_x, pixel_length_y, valid_mask=valid_mask)

            all_results[(min_move, max_move)].append({
                "dataset": dataset,
                "subject": subject.id,
                "comparison": subject.comparison_label,
                "min_movement": min_move,
                "max_movement": max_move,
                **metrics,
            })

            if save_figures:
                fig_path = os.path.join(figures_path, f"{subject.id}_{mode}_{tag}.png")
                save_quiver_figure(subject, u, v, zones, inner_radius_x, inner_radius_y, outer_radius_x, outer_radius_y, fig_path)

    results_by_threshold = {}
    for threshold, rows in all_results.items():
        tag = _threshold_tag(*threshold)
        results_df = pd.DataFrame(rows)
        csv_path = os.path.join(results_data_path, f"dataset{dataset}_{mask_source}_{mode}_{tag}_metrics.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\nMetriche salvate in: {csv_path}")
        results_by_threshold[threshold] = results_df

    return results_by_threshold


def _parse_threshold(raw: str) -> tuple[float, float]:
    try:
        min_str, max_str = raw.split("-")
        min_move, max_move = float(min_str), float(max_str)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Soglia '{raw}' non valida: usa il formato min-max (es. 1-10)"
        )
    if min_move >= max_move:
        raise argparse.ArgumentTypeError(f"Soglia '{raw}' non valida: min deve essere minore di max")
    return min_move, max_move


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=int, choices=[1, 2], required=True, help="Dataset su cui calcolare l'optical flow")
    parser.add_argument("--mode", choices=["interpolated", "masked"], required=True,
                         help="'interpolated' = interpolazione RBF sui vasi; 'masked' = solo sui pixel dei vasi, senza interpolazione")
    parser.add_argument("--mask-source", choices=["gt", "auto"], default="gt",
                         help="'gt' = maschere ground-truth, su tutti i soggetti (default); "
                              "'auto' = maschere di segmentazione automatica, solo dataset 1, limitate ai 10 soggetti con Dice Score migliore")
    parser.add_argument("--save-figures", action="store_true", help="Salva anche le figure con il campo vettoriale sovrapposto")
    parser.add_argument("--thresholds", nargs="+", type=_parse_threshold, default=None, metavar="MIN-MAX",
                         help="Una o più coppie soglia_min-soglia_max (es. --thresholds 1-10 2-15 3-20). "
                              "Ogni coppia produce un CSV separato. Se omesso usa MOVEMENT_THRESHOLDS da config.py")
    args = parser.parse_args()

    run(dataset=args.dataset, mode=args.mode, mask_source=args.mask_source,
        save_figures=args.save_figures, thresholds=args.thresholds)
