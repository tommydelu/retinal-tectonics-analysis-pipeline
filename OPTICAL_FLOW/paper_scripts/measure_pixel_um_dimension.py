#------------------------------ LIBRARIES ------------------------------#

import os
import cv2 as cv
import numpy

from common.paths import PROJECT_ROOT

#-----------------------------------------------------------------------#

"""
Idea: isolare uno dei 4 angoli dell'immagine per centrare il tutto sulla barretta che definisce la scala.
Poi sfruttare il fatto che la barretta è completamente bianca (valore 255) e sfruttando i contorni trovare
la sua bounding box e quindi la larghezza in pixel.
"""

# since in ds2 i have different resolutions, for each resolution i will have to define the pixel to um conversion factor.
bar_length_mapping = {"1904,1912": None, "1398,1399": None, "1906,1912": None, "1395,1330": None, "1417,1421": None,
                      "1674,1681": None, "1676,1681": None, "1678,1678": None, "1398,1398": None, "1904,1911": None,
                      "1489,1473": None, "1414,1409": None, "1357,1357": None, "1358,1358": None, "1363,1357": None,
                      "1415,1334": None, "1536,1636": None, "1679,1679": None, "1363,1363": None, "1680,1680": None,
                      "1492,1480": None, "1427,1393": None, "1449,1439": None, "943,1005": None, "1412,1429": None,
                      "1364,1364": None}

ds2_path = os.path.join(PROJECT_ROOT, 'DATA', 'DATASET2', 'raw', 'Immagini_IR')

# Percentuali che definiscono quanto è "grande" la regione d'angolo da ispezionare,
# come frazione delle dimensioni dell'immagine.
ROI_H_PCT = 0.12   # altezza della ROI (es. ultimo/primo 12% dell'immagine)
ROI_W_PCT = 0.15   # larghezza della ROI (es. primi/ultimi 15% dell'immagine)

# Soglia minima di aspect ratio (w/h) per considerare un contorno come "barretta"
# (la barra della scala è tipicamente molto più larga che alta)
MIN_ASPECT_RATIO = 3.0

DEBUG_SHOW = True  # metti a False per disattivare tutte le imshow di debug

with open("nervo_a_sinistra.txt", 'r') as f:
    names = [line.strip() for line in f.readlines()]


def get_corner_rois(img_gray):
    """
    Ritorna le 4 ROI d'angolo dell'immagine e i relativi offset (x, y)
    rispetto all'immagine originale, utili per rimappare i contorni trovati.
    """
    h, w = img_gray.shape[:2]

    roi_h = int(h * ROI_H_PCT)
    roi_w = int(w * ROI_W_PCT)

    corners = {
        "top_left":     img_gray[0:roi_h,       0:roi_w],
        "top_right":    img_gray[0:roi_h,       w - roi_w:w],
        "bottom_left":  img_gray[h - roi_h:h,   0:roi_w],
        "bottom_right": img_gray[h - roi_h:h,   w - roi_w:w],
    }

    offsets = {
        "top_left":     (0,         0),
        "top_right":    (w - roi_w, 0),
        "bottom_left":  (0,         h - roi_h),
        "bottom_right": (w - roi_w, h - roi_h),
    }

    return corners, offsets


def find_bar_in_corner(roi_gray):
    """
    Cerca la barretta bianca in una ROI d'angolo.
    Ritorna (contour, bounding_rect) se trovata con aspect ratio valido, altrimenti None.
    """
    _, thresh = cv.threshold(roi_gray, 240, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # ordina i contorni per area decrescente, in caso il più grande non sia quello giusto
    contours = sorted(contours, key=cv.contourArea, reverse=True)

    for cnt in contours:
        x, y, w_bar, h_bar = cv.boundingRect(cnt)
        if h_bar == 0:
            continue
        aspect_ratio = w_bar / h_bar
        if aspect_ratio >= MIN_ASPECT_RATIO:
            return cnt, (x, y, w_bar, h_bar)

    return None


def process_images(dir_path, bar_length_mapping, names):

    for f in sorted(os.listdir(dir_path)):

        name = f.split('.')[0].split('IR')[0]

        img_path = os.path.join(dir_path, f)
        img_gray = cv.imread(img_path, 0)

        if img_gray is None:
            print(f"Impossibile leggere {img_path}, skip.")
            continue

        if name not in names:
            img_gray = cv.flip(img_gray, 1)

        h, w = img_gray.shape[:2]
        res_key = f"{w},{h}"

        if res_key not in bar_length_mapping:
            print(f"Risoluzione {res_key} non presente nel dizionario, skip. ({f})")
            continue
        if bar_length_mapping[res_key] is not None:
            continue  # già trovata per questa risoluzione

        img_bgr = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)

        corners, offsets = get_corner_rois(img_gray)

        found = False
        for corner_name, roi_gray in corners.items():
            x_offset, y_offset = offsets[corner_name]

            if DEBUG_SHOW:
                cv.imshow(f"roi_{corner_name}", roi_gray)
                cv.waitKey(0)
                cv.destroyWindow(f"roi_{corner_name}")

            result = find_bar_in_corner(roi_gray)
            if result is None:
                continue

            cnt, (x, y, w_bar, h_bar) = result
            cnt_offset = cnt + [x_offset, y_offset]
            cv.drawContours(img_bgr, [cnt_offset], -1, (0, 0, 255), 2)

            print(f"[{res_key}] Barra trovata in '{corner_name}': {w_bar} pixel ({f})")
            bar_length_mapping[res_key] = w_bar
            found = True
            break  # trovata, non serve controllare gli altri corner

        if not found:
            print(f"[{res_key}] Nessuna barra trovata in nessun corner ({f})")

        if DEBUG_SHOW:
            cv.imshow('Immagine con identificata la barretta per la scala', img_bgr)
            cv.waitKey(0)


process_images(ds2_path, bar_length_mapping, names)

cv.destroyAllWindows()

print(bar_length_mapping)

# eventuale controllo finale su quali risoluzioni non sono state trovate
missing = [k for k, v in bar_length_mapping.items() if v is None]
if missing:
    print("\n")
    print(f"Attenzione: nessuna barra trovata per le risoluzioni: {missing}")