import os
import json
import glob
from dataclasses import dataclass
from typing import Optional, Iterator

import cv2 as cv
import numpy as np

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
    Soggetti del dataset 1: confronto PRE/POST.

    mask_source="gt" (default): maschera dei vasi dalle label manuali (ground-truth),
    su TUTTI i soggetti del dataset.
    mask_source="auto": maschera dai risultati della segmentazione automatica, limitata
    ai 10 soggetti con Dice Score migliore (BEST_SUBJECTS) — è l'unico sottoinsieme per
    cui la segmentazione automatica è stata generata/validata.

    Il centro della fovea è noto a priori per i 10 BEST_SUBJECTS (cliccato manualmente
    una volta). Per tutti gli altri soggetti (rilevanti solo con mask_source="gt") viene
    richiesto con un click interattivo al primo utilizzo e da lì in poi riusato da una
    cache su file JSON, con lo stesso meccanismo di Dataset2Subjects.

    La maschera dei vasi (gt o auto) viene flippata orizzontalmente per i soggetti
    elencati in vasi_specchiati_dataset1.txt (nella root del progetto), generato da
    OPTICAL_FLOW/segmentation/src/utils/extract_flipped_subjects.py a partire dalla
    colonna 'Flipped' di seg_metrics1.csv — cioè i soggetti per cui l'immagine IR
    risulta specchiata rispetto ai vasi annotati.
    """

    BEST_SUBJECTS = ['L_03', 'L_06', 'L_15', 'L_16', 'L_42', 'L_48', 'L_63', 'L_78', 'S_36', 'S_46']

    FOVEA_CENTER_DICT = {
        'L_03': (1013, 1006), 'L_06': (968, 1078), 'L_15': (1037, 961), 'L_16': (968, 985),
        'L_42': (980, 1011), 'L_48': (932, 934), 'L_63': (1042, 961), 'L_78': (1016, 937),
        'S_36': (968, 1006), 'S_46': (853, 1120),
    }

    def __init__(self, mask_source: str = "gt"):
        if mask_source not in ("gt", "auto"):
            raise ValueError(f"mask_source deve essere 'gt' o 'auto', ricevuto: {mask_source!r}")
        self.mask_source = mask_source

        self.src_path = os.path.join(PROJECT_ROOT, 'DATA', 'DATASET1', 'raw', 'IR')
        self.labels_path = os.path.join(PROJECT_ROOT, 'DATA', 'DATASET1', 'processed','labels_reversed')
        self.auto_masks_path = os.path.join(PROJECT_ROOT, 'OPTICAL_FLOW', 'segmentation', 'results', 'figures', 'mergedVessels_1')

        flipped_vessels_path = os.path.join(PROJECT_ROOT, 'vasi_specchiati_dataset1.txt')
        self._flipped_subjects = set()
        if os.path.exists(flipped_vessels_path):
            with open(flipped_vessels_path, 'r') as f:
                self._flipped_subjects = {line.strip() for line in f if line.strip()}

        results_data_path = os.path.join(PROJECT_ROOT, 'OPTICAL_FLOW', 'farneback', 'results', 'ds1', 'data')
        self.fovea_json_path = os.path.join(results_data_path, 'fovea_centers.json')
        os.makedirs(results_data_path, exist_ok=True)

    def _is_flipped(self, subject: str) -> bool:
        return subject in self._flipped_subjects

    @staticmethod
    def _resize_to_match(mask: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Alcuni soggetti hanno maschera/label a una risoluzione diversa dall'immagine IR."""
        if mask.shape != target_shape:
            mask = cv.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv.INTER_NEAREST)
        return mask

    def _find_auto_mask(self, subject: str, timepoint: str) -> Optional[np.ndarray]:
        for ext in ('.JPG', '.jpg', '.JPEG', '.jpeg', '.PNG', '.png'):
            path = os.path.join(self.auto_masks_path, f"{subject}{timepoint}{ext}")
            if os.path.exists(path):
                return cv.imread(path, 0)
        return None

    def _get_fovea_center(self, subject: str, display_img: np.ndarray) -> tuple:
        if subject in self.FOVEA_CENTER_DICT:
            return self.FOVEA_CENTER_DICT[subject]

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

        for fname in sorted(os.listdir(self.src_path)):

            if 'POST' in fname:
                continue  # si processa la coppia PRE-POST una volta sola, all'incontro del PRE

            subject = fname.split('PRE')[0]
            if self.mask_source == "auto" and subject not in self.BEST_SUBJECTS:
                continue

            fname_post = f"{subject}POST.JPG"

            img_pre_raw = cv.imread(os.path.join(self.src_path, fname), 0)
            img_post_raw = cv.imread(os.path.join(self.src_path, fname_post), 0)
            img_pre = clahe(img_pre_raw, 2, 4)
            img_post = clahe(img_post_raw, 2, 4)

            if self.mask_source == "gt":
                vessel_mask_pre = cv.imread(os.path.join(self.labels_path, subject, 'total_1.png'), 0)
                vessel_mask_post = cv.imread(os.path.join(self.labels_path, subject, 'total_2.png'), 0)
                if vessel_mask_pre is None or vessel_mask_post is None:
                    continue
                vessel_mask_pre = self._resize_to_match(vessel_mask_pre, img_pre_raw.shape)
                vessel_mask_post = self._resize_to_match(vessel_mask_post, img_post_raw.shape)
                if self._is_flipped(subject):
                    vessel_mask_pre = cv.flip(vessel_mask_pre, 1)
                    vessel_mask_post = cv.flip(vessel_mask_post, 1)
                vessel_mask_pre = vessel_mask_pre > 0
                vessel_mask_post = vessel_mask_post > 0
            else:  # auto
                vessel_mask_pre = self._find_auto_mask(subject, 'PRE')
                vessel_mask_post = self._find_auto_mask(subject, 'POST')
                if vessel_mask_pre is None or vessel_mask_post is None:
                    continue
                vessel_mask_pre = self._resize_to_match(vessel_mask_pre, img_pre_raw.shape)
                vessel_mask_post = self._resize_to_match(vessel_mask_post, img_post_raw.shape)
                # JPG con lieve rumore di compressione attorno a 0/255: soglia a metà scala
                vessel_mask_pre = vessel_mask_pre > 127
                vessel_mask_post = vessel_mask_post > 127

            fovea_center = self._get_fovea_center(subject, img_pre)

            yield SubjectPair(
                id=subject,
                comparison_label="PRE-POST",
                img_pre=img_pre,
                img_post=img_post,
                vessel_mask_pre=vessel_mask_pre,
                vessel_mask_post=vessel_mask_post,
                fovea_center=fovea_center,
                draw_image=cv.cvtColor(img_pre_raw, cv.COLOR_GRAY2BGR),
            )


class Dataset2Subjects:
    """
    Soggetti del dataset 2: confronto baseline (mese 0) vs 12 mesi post-operatorio,
    su tutti i soggetti disponibili. La fovea viene selezionata manualmente al primo
    utilizzo (click interattivo) e da lì in poi riusata da una cache su file JSON.
    """

    FOLLOW_UP = '1'

    def __init__(self):
        self.src_path = os.path.join(PROJECT_ROOT, 'DATA', 'DATASET2', 'raw', 'Immagini_IR_Corrette')
        self.vessels_base_path = os.path.join(PROJECT_ROOT, 'DATA', 'DATASET2', 'raw', 'Vessels_def')
        results_data_path = os.path.join(PROJECT_ROOT, 'OPTICAL_FLOW', 'farneback', 'results', 'ds2', 'data')
        self.fovea_json_path = os.path.join(results_data_path, 'fovea_centers.json')
        os.makedirs(results_data_path, exist_ok=True)

        left_eyes_file = os.path.join(PROJECT_ROOT, 'nervo_a_sinistra.txt')
        self.flip_list = []
        if os.path.exists(left_eyes_file):
            with open(left_eyes_file, 'r') as f:
                self.flip_list = [line.strip().lower().replace("_", "") for line in f if line.strip()]

    def match_shape(img: np.ndarray, target_shape: tuple[int, int], tolerance: int = 20) -> np.ndarray:
        """
        Uniforma la forma di 'img' a target_shape=(target_h, target_w).
        - Se la differenza dimensionale è <= tolerance, usa crop/pad centrato per preservare la calibrazione.
        - Se la differenza è > tolerance, assume un reale cambio di scala e applica cv.resize.
        """
        target_h, target_w = target_shape
        h, w = img.shape[:2]
        
        if (h, w) == (target_h, target_w):
            return img

        # --- CONTROLLO AUTOMATICO: Crop/Pad o Resize? ---
        if abs(h - target_h) > tolerance or abs(w - target_w) > tolerance:
            # Differenza ampia: usiamo il resize interpolato
            # print(f"  [Info] Shape mismatch elevato ({w}x{h} -> {target_w}x{target_h}). Applico cv.resize.")
            # Se è una maschera binaria (es. label vasi), meglio INTER_NEAREST per non sfuocare i bordi
            interp = cv.INTER_NEAREST if img.dtype == bool or len(np.unique(img)) <= 2 else cv.INTER_LINEAR
            return cv.resize(img, (target_w, target_h), interpolation=interp)

        # --- DIFFERENZA MINIMA: Usiamo Crop / Pad ---
        # 1. Croppa se l'immagine è più grande
        if h > target_h:
            diff_y = h - target_h
            top = diff_y // 2
            img = img[top:top + target_h, :]
        if w > target_w:
            diff_x = w - target_w
            left = diff_x // 2
            img = img[:, left:left + target_w]

        # 2. Aggiunge bordi neri se l'immagine è diventata più piccola (o lo era già)
        h, w = img.shape[:2]
        pad_top = max(0, (target_h - h) // 2)
        pad_bottom = max(0, target_h - h - pad_top)
        pad_left = max(0, (target_w - w) // 2)
        pad_right = max(0, target_w - w - pad_left)

        if any((pad_top, pad_bottom, pad_left, pad_right)):
            img = cv.copyMakeBorder(
                img, pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv.BORDER_CONSTANT, value=0
            )

        return img
    
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
            img_pre_raw = self._get_ir_image(subject, '0')
            if img_pre_raw is None:
                continue

            h, w = img_pre_raw.shape
            fovea_center = self._get_fovea_center(subject, clahe(img_pre_raw, 2, 4))

            label_pre = self._get_vessel_label(subject, '0')
            if label_pre is None:
                continue
            # Uniforma la maschera pre alla baseline
            if label_pre.shape != (h, w):
                label_pre = self.match_shape(label_pre, (h, w))

            img_post = self._get_ir_image(subject, self.FOLLOW_UP)
            if img_post is None:
                continue
            # Uniforma img_post alla baseline senza stirare l'immagine
            if img_post.shape != (h, w):
                img_post = self.match_shape(img_post, (h, w))
            img_post = clahe(img_post, 2, 4)

            label_post = self._get_vessel_label(subject, self.FOLLOW_UP)
            if label_post is None:
                continue
            # Uniforma la maschera post alla baseline
            if label_post.shape != (h, w):
                label_post = self.match_shape(label_post, (h, w))

            img_pre = clahe(img_pre_raw, 2, 4)

            yield SubjectPair(
                id=subject,
                comparison_label=f"0-{self.FOLLOW_UP}",
                img_pre=img_pre,
                img_post=img_post,
                vessel_mask_pre=self._label_to_vessel_mask(label_pre),
                vessel_mask_post=self._label_to_vessel_mask(label_post),
                fovea_center=fovea_center,
                draw_image=cv.cvtColor(img_pre_raw, cv.COLOR_GRAY2BGR),
            )
