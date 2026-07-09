import os
import json
import glob
from dataclasses import dataclass
from typing import Optional, Iterator

import cv2 as cv
import numpy as np
import pandas as pd

from common.paths import PROJECT_ROOT
from common.image_filters import clahe


@dataclass
class SubjectPair:
    """Una coppia di immagini PRE/POST pronta per il calcolo dell'optical flow."""
    id: str
    comparison_label: str
    img_pre: np.ndarray
    img_post: np.ndarray
    vessel_mask_pre: np.ndarray
    vessel_mask_post: np.ndarray
    fovea_center: tuple
    draw_image: Optional[np.ndarray] = None


class Dataset1Subjects:
    """
    Soggetti del dataset 1: confronto PRE/POST su un set fisso di 10 soggetti già
    selezionati, con fovea nota a priori e maschera dei vasi derivata dalle label
    manuali (ground-truth), non dall'output della segmentazione automatica.
    """

    BEST_SUBJECTS = ['L_03', 'L_06', 'L_15', 'L_16', 'L_42', 'L_48', 'L_63', 'L_78', 'S_36', 'S_46']

    FOVEA_CENTER_DICT = {
        'L_03': (1013, 1006), 'L_06': (968, 1078), 'L_15': (1037, 961), 'L_16': (968, 985),
        'L_42': (980, 1011), 'L_48': (932, 934), 'L_63': (1042, 961), 'L_78': (1016, 937),
        'S_36': (968, 1006), 'S_46': (853, 1120),
    }

    def __init__(self):
        self.src_path = os.path.join(PROJECT_ROOT, 'DATA', 'DATASET1', 'raw', 'IR')
        self.labels_path = os.path.join(PROJECT_ROOT, 'DATA', 'DATASET1', 'processed', 'results', 'figures', 'labels_reversed')
        self.csv_seg_path = os.path.join(PROJECT_ROOT, 'OPTICAL_FLOW', 'segmentation', 'results', 'data', 'seg_metrics1.csv')

    def _is_flipped(self, subject: str) -> bool:
        flipped_cols_csv = pd.read_csv(self.csv_seg_path, index_col=False, usecols=['SUBJECT', 'Flipped'])
        row = flipped_cols_csv.loc[flipped_cols_csv['SUBJECT'] == f"{subject}PRE", 'Flipped']
        return bool(row.values[0])

    def __iter__(self) -> Iterator[SubjectPair]:
        for fname in sorted(os.listdir(self.src_path)):
            if 'POST' in fname:
                continue  # si processa la coppia PRE-POST una volta sola, all'incontro del PRE

            subject = fname.split('PRE')[0]
            if subject not in self.BEST_SUBJECTS:
                continue

            fname_post = f"{subject}POST.JPG"

            img_pre = clahe(cv.imread(os.path.join(self.src_path, fname), 0), 2, 4)
            img_post = clahe(cv.imread(os.path.join(self.src_path, fname_post), 0), 2, 4)

            vessel_mask_pre = cv.imread(os.path.join(self.labels_path, subject, 'total_1.png'), 0)
            vessel_mask_post = cv.imread(os.path.join(self.labels_path, subject, 'total_2.png'), 0)

            if self._is_flipped(subject):
                vessel_mask_pre = cv.flip(vessel_mask_pre, 1)
                vessel_mask_post = cv.flip(vessel_mask_post, 1)

            yield SubjectPair(
                id=subject,
                comparison_label="PRE-POST",
                img_pre=img_pre,
                img_post=img_post,
                vessel_mask_pre=vessel_mask_pre > 0,
                vessel_mask_post=vessel_mask_post > 0,
                fovea_center=self.FOVEA_CENTER_DICT[subject],
                draw_image=cv.cvtColor(img_pre, cv.COLOR_GRAY2BGR),
            )


class Dataset2Subjects:
    """
    Soggetti del dataset 2: confronto baseline (mese 0) vs 12 mesi post-operatorio,
    su tutti i soggetti disponibili. La fovea viene selezionata manualmente al primo
    utilizzo (click interattivo) e da lì in poi riusata da una cache su file JSON.
    """

    FOLLOW_UP = '12'

    def __init__(self):
        self.src_path = os.path.join(PROJECT_ROOT, 'DATA', 'DATASET2', 'raw', 'Immagini_IR')
        self.vessels_base_path = os.path.join(PROJECT_ROOT, 'DATA', 'DATASET2', 'raw', 'Vessels_def')
        results_path = os.path.join(PROJECT_ROOT, 'OPTICAL_FLOW', 'results', 'ds2', 'farneback', 'results')
        self.fovea_json_path = os.path.join(results_path, 'data', 'fovea_centers.json')
        os.makedirs(os.path.dirname(self.fovea_json_path), exist_ok=True)

        left_eyes_file = os.path.join(PROJECT_ROOT, 'nervo_a_sinistra.txt')
        self.flip_list = []
        if os.path.exists(left_eyes_file):
            with open(left_eyes_file, 'r') as f:
                self.flip_list = [line.strip().lower().replace("_", "") for line in f if line.strip()]

    def _discover_subjects(self) -> list[str]:
        baseline_files = glob.glob(os.path.join(self.src_path, "*_0.*"))
        return sorted({os.path.basename(f).split('_0')[0] for f in baseline_files if '_0' in f})

    def _get_ir_image(self, subject: str, timepoint: str) -> Optional[np.ndarray]:
        possible_files = glob.glob(os.path.join(self.src_path, f"{subject}_{timepoint}.*"))
        valid_extensions = ['.png', '.jpg', '.jpeg']

        for file_path in possible_files:
            if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
                continue
            img = cv.imread(file_path, 0)
            if img is None:
                return None
            subj_clean = subject[:-2].lower().replace("_", "") if subject.endswith('IR') else subject.lower().replace("_", "")
            if any(left_eye in subj_clean for left_eye in self.flip_list):
                img = cv.flip(img, 1)
            return img
        return None

    def _get_vessel_label(self, subject: str, timepoint: str) -> Optional[np.ndarray]:
        label_subject = subject[:-2] if subject.endswith('IR') else subject
        subj_clean = label_subject.lower().replace("_", "")
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif']
        target_suffix = f"_{timepoint}."

        for root, _, files in os.walk(self.vessels_base_path):
            for file in files:
                file_lower = file.lower()
                if target_suffix not in file_lower or not any(file_lower.endswith(ext) for ext in valid_extensions):
                    continue
                base_name_mask = file_lower.split(target_suffix)[0].replace("_", "")
                if subj_clean.startswith(base_name_mask):
                    return cv.imread(os.path.join(root, file), 0)
        return None

    @staticmethod
    def _label_to_vessel_mask(label_img: np.ndarray) -> np.ndarray:
        """Le label del dataset 2 possono avere sfondo chiaro o scuro: si sceglie la soglia di conseguenza."""
        bg_value = np.median(label_img)
        return (label_img < 245) if bg_value > 127 else (label_img > 10)

    def _get_fovea_center(self, subject: str, display_img: np.ndarray) -> tuple:
        fovea_dict = {}
        if os.path.exists(self.fovea_json_path):
            with open(self.fovea_json_path, 'r') as f:
                fovea_dict = json.load(f)

        if subject in fovea_dict:
            return tuple(fovea_dict[subject])

        center = []

        def click_event(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                center.append((x, y))

        window_name = f"Seleziona Fovea per {subject} (Clicca e poi premi un tasto)"
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.setMouseCallback(window_name, click_event)

        display = display_img.copy()
        cv.putText(display, "Clicca il centro della fovea e premi un tasto", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv.imshow(window_name, display)
        cv.waitKey(0)
        cv.destroyAllWindows()

        chosen_center = center[0] if center else (0, 0)
        fovea_dict[subject] = chosen_center
        with open(self.fovea_json_path, 'w') as f:
            json.dump(fovea_dict, f)

        return tuple(chosen_center)

    def __iter__(self) -> Iterator[SubjectPair]:
        for subject in self._discover_subjects():
            img_pre = self._get_ir_image(subject, '0')
            if img_pre is None:
                continue

            h, w = img_pre.shape
            fovea_center = self._get_fovea_center(subject, clahe(img_pre, 2, 4))

            label_pre = self._get_vessel_label(subject, '0')
            if label_pre is None:
                continue
            if label_pre.shape != (h, w):
                label_pre = cv.resize(label_pre, (w, h), interpolation=cv.INTER_NEAREST)

            img_post = self._get_ir_image(subject, self.FOLLOW_UP)
            if img_post is None:
                continue
            if img_post.shape != (h, w):
                img_post = cv.resize(img_post, (w, h), interpolation=cv.INTER_LINEAR)
            img_post = clahe(img_post, 2, 4)

            label_post = self._get_vessel_label(subject, self.FOLLOW_UP)
            if label_post is None:
                continue
            if label_post.shape != (h, w):
                label_post = cv.resize(label_post, (w, h), interpolation=cv.INTER_NEAREST)

            yield SubjectPair(
                id=subject,
                comparison_label=f"0-{self.FOLLOW_UP}",
                img_pre=img_pre,
                img_post=img_post,
                vessel_mask_pre=self._label_to_vessel_mask(label_pre),
                vessel_mask_post=self._label_to_vessel_mask(label_post),
                fovea_center=fovea_center,
                draw_image=cv.cvtColor(img_pre, cv.COLOR_GRAY2BGR),
            )
