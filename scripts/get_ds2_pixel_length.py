
import os
import cv2 as cv
import numpy as np
import math
import csv
from common.paths import PROJECT_ROOT
import re

# --- PATH DELLE CARTELLE ---
DS2_DIR_PATH1 = os.path.join(PROJECT_ROOT, 'DATA', 'DATASET2', 'raw', 'Immagini_IR_Corrette')
DS2_DIR_PATH2 = os.path.join(PROJECT_ROOT, 'DATA', 'DATASET2', 'raw', 'IR_Def2')
DS2_DIR_PATH3 = os.path.join(PROJECT_ROOT, 'DATA', 'DATASET2', 'raw', 'IR_Def')

CSV_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'scala_post_registrazione.csv')

SOGGETTI_SX = [
    "albanese", "casale", "cortellessa", "desantis", "de_santis", "difede", 
    "di_fede", "fasciano", "ferrara", "giamundo", "inglese", "limongi", 
    "manganiello", "nave", "rinaldi"
]

clicked_points = []
img_display = None

def mouse_callback(event, x, y, flags, param):
    global clicked_points, img_display
    if event == cv.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 2:
            clicked_points.append((x, y))
            cv.circle(img_display, (x, y), 5, (0, 0, 255), -1)
            if len(clicked_points) == 2:
                cv.line(img_display, clicked_points[0], clicked_points[1], (0, 255, 0), 2)
            cv.imshow("Selezione Landmark", img_display)



def find_scale_bar_px(img, subject_name, debug=True):
    h, w = img.shape[:2]
    
    roi_h = min(50, h // 4)
    roi_w = min(50, w // 4)
    
    roi_left_rect = (0, h - roi_h, roi_w, roi_h)
    roi_left = img[h-roi_h:h, 0:roi_w]
    
    roi_right_rect = (w - roi_w, h - roi_h, roi_w, roi_h)
    roi_right = img[h-roi_h:h, w-roi_w:w]
    
    start_left = any(sx_name in subject_name.lower() for sx_name in SOGGETTI_SX)
    
    if start_left:
        rois_to_check = [("Sinistra", roi_left, roi_left_rect), ("Destra", roi_right, roi_right_rect)]
    else:
        rois_to_check = [("Destra", roi_right, roi_right_rect), ("Sinistra", roi_left, roi_left_rect)]
        
    for side_name, roi, roi_rect in rois_to_check:
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 220, 255, cv.THRESH_BINARY)

        # Dilatazione leggera per ricucire eventuali micro-interruzioni della "L"
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        thresh_dilatato = cv.dilate(thresh, kernel, iterations=1)

        if debug:
            scale_vis = 6
            roi_grande = cv.resize(roi, (roi.shape[1]*scale_vis, roi.shape[0]*scale_vis),
                                    interpolation=cv.INTER_NEAREST)
            thresh_grande = cv.resize(thresh, (thresh.shape[1]*scale_vis, thresh.shape[0]*scale_vis),
                                       interpolation=cv.INTER_NEAREST)
            dilatato_grande = cv.resize(thresh_dilatato, (thresh_dilatato.shape[1]*scale_vis, thresh_dilatato.shape[0]*scale_vis),
                                         interpolation=cv.INTER_NEAREST)
            print(f"[{subject_name}] Visualizzazione ROI lato {side_name.upper()} "
                  f"(premi un tasto per continuare...)")
            cv.imshow(f"1 - ROI originale ({side_name})", roi_grande)
            cv.imshow(f"3 - Soglia/Threshold ({side_name})", thresh_grande)
            cv.imshow(f"4 - Dilatata ({side_name})", dilatato_grande)
            cv.waitKey(0)
            cv.destroyAllWindows()

        contours, _ = cv.findContours(thresh_dilatato, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue

        # --- Isoliamo la "L": bounding box con dimensioni comparabili, non troppo piccola ---
        candidati_L = []
        for cnt in contours:
            x, y, cw, ch = cv.boundingRect(cnt)
            if debug:
                aspect_ratio = cw / float(ch) if ch > 0 else 0
                print(f"    contorno: cw={cw} ch={ch} aspect={aspect_ratio:.2f}")

            if cw < 8 or ch < 8:
                continue
            if cw > roi_w * 0.9 or ch > roi_h * 0.9:
                continue

            candidati_L.append((x, y, cw, ch))

        if not candidati_L:
            continue

        # Prendiamo il candidato con area maggiore (la "L" è l'oggetto più esteso rimasto)
        x, y, cw, ch = max(candidati_L, key=lambda c: c[2] * c[3])

        # --- Misuriamo il tratto ORIZZONTALE della "L" sul thresh non dilatato ---
        sotto_regione = thresh[y:y+ch, x:x+cw]
        migliore_larghezza = 0
        for riga in range(sotto_regione.shape[0]):
            pixel_riga = sotto_regione[riga, :]
            bianchi = np.where(pixel_riga > 0)[0]
            if len(bianchi) > 0:
                larghezza_riga = bianchi[-1] - bianchi[0] + 1
                migliore_larghezza = max(migliore_larghezza, larghezza_riga)

        bar_width_px = migliore_larghezza

        if bar_width_px < 5:
            continue  # troppo corto per essere plausibile, scarta

        if debug:
            print(f"[{subject_name}] -> 'L' trovata a {side_name.upper()}! "
                  f"Lato orizzontale: {bar_width_px}px")
            roi_color = roi.copy()
            cv.rectangle(roi_color, (x, y), (x + cw, y + ch), (0, 255, 0), 1)
            roi_color_grande = cv.resize(roi_color, (roi_color.shape[1]*6, roi_color.shape[0]*6),
                                          interpolation=cv.INTER_NEAREST)
            cv.imshow(f"Debug Finale ROI - {subject_name}", roi_color_grande)
            cv.waitKey(0)
            cv.destroyAllWindows()
            
        return bar_width_px
        
    return 0


def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


import difflib

def normalizza_nome(nome):
    """Normalizza un nome soggetto: minuscolo, toglie 'ir' e caratteri non alfabetici.
    NON rimuove più singole lettere comuni (era la causa dei falsi match)."""
    nome = nome.lower()
    nome = re.sub(r'\bir\b', '', nome)   # rimuove "ir" come parola/suffisso
    nome = nome.replace('ir', '')        # rimuove eventuale "ir" attaccato (es. "cognomeir")
    nome = re.sub(r'[^a-z]', '', nome)   # tiene solo lettere
    return nome


def find_pre_image(subject_prefix, soglia_similitudine=0.85):
    """Cerca l'immagine PRE corrispondente, con controllo di corrispondenza sul nome
    per evitare di associare soggetti diversi."""
    target_norm = normalizza_nome(subject_prefix)

    candidati = []  # (score, path, fname)

    for folder in [DS2_DIR_PATH2, DS2_DIR_PATH3]:
        if not os.path.exists(folder):
            continue
        for root, dirs, files in os.walk(folder):
            for fname in files:
                fname_lower = fname.lower()
                if not ("preop" in fname_lower or "baseline" in fname_lower or "_0." in fname_lower):
                    continue

                # normalizza anche il nome del file, rimuovendo la parte numerica/suffissi noti
                base_fname = re.sub(r'(preop|baseline|_0\.(png|jpg|jpeg))', '', fname_lower)
                fname_norm = normalizza_nome(base_fname)

                if not fname_norm:
                    continue

                # Match esatto sul nome normalizzato: massima priorità
                if fname_norm == target_norm:
                    score = 1.0
                else:
                    # match parziale (uno contenuto nell'altro) + similarità testuale
                    contenuto = target_norm in fname_norm or fname_norm in target_norm
                    ratio = difflib.SequenceMatcher(None, target_norm, fname_norm).ratio()
                    if not contenuto and ratio < soglia_similitudine:
                        continue
                    score = ratio

                candidati.append((score, os.path.join(root, fname), fname))

    if not candidati:
        return None

    # Ordina per punteggio decrescente e prendi il migliore
    candidati.sort(key=lambda x: x[0], reverse=True)
    best_score, best_path, best_fname = candidati[0]

    # --- CONTROLLO ANTI-AMBIGUITA' ---
    # Se ci sono più candidati con punteggio molto vicino al migliore, è ambiguo: meglio segnalarlo
    ambigui = [c for c in candidati if c[0] >= best_score - 0.05 and c[1] != best_path]
    if ambigui:
        print(f"[{subject_prefix}] ATTENZIONE: match ambiguo tra più file simili:")
        print(f"   -> Scelto: {best_fname} (score={best_score:.2f})")
        for score, path, fname in ambigui:
            print(f"   -> Alternativa scartata: {fname} (score={score:.2f})")

    return best_path



def main():
    global clicked_points, img_display
    
    with open(CSV_OUTPUT_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Soggetto', 'Scala_PRE_um_px', 'Scala_POST_um_px'])

    post_files = sorted(os.listdir(DS2_DIR_PATH1))
    post_baselines = [f for f in post_files if f.endswith('_0.png') or f.endswith('_0.JPG')]

    for post_fname in post_baselines:
        subject_name = post_fname.replace('_0.png', '').replace('_0.JPG', '')
        
        post_path = os.path.join(DS2_DIR_PATH1, post_fname)
        pre_path = find_pre_image(subject_name)
        
        if not pre_path:
            print(f"[{subject_name}] ERRORE: Immagine PRE non trovata. Salto...")
            continue

        # --- CONTROLLO ESPLICITO SUL NOME PRIMA DI PROCEDERE ---
        print(f"\n[{subject_name}] Soggetto POST : {post_fname}")
        print(f"[{subject_name}] Soggetto PRE  : {os.path.basename(pre_path)}")
        conferma = input("Le due immagini corrispondono allo stesso soggetto? (s/n): ").strip().lower()
        if conferma != 's':
            print(f"[{subject_name}] Saltato per conferma negativa dell'utente.\n")
            continue
            
        pre_img = cv.imread(pre_path)
        post_img = cv.imread(post_path)

        is_left = any(sx_name in subject_name.lower() for sx_name in SOGGETTI_SX)
        
        # --- 1. TROVA LA SCALA PRE (Con Debug) ---
        bar_px = find_scale_bar_px(pre_img, subject_name, debug=True)
        if bar_px == 0:
            print(f"[{subject_name}] ATTENZIONE: Scale bar non rilevata. Salto al prossimo...")
            continue
            
        pre_scale = 200.0 / bar_px
        print(f"\n--- Elaborazione: {subject_name} ---")
        print(f"Barra trovata: {bar_px} px -> Scala PRE: {pre_scale:.3f} um/px")
        
        cv.namedWindow("Selezione Landmark", cv.WINDOW_NORMAL)
        cv.resizeWindow("Selezione Landmark", 900, 900)

        # --- 2. CLICCA SULLA PRE ---
        while True:
            clicked_points = []
            img_display = pre_img.copy()
            cv.imshow("Selezione Landmark", img_display)
            cv.setMouseCallback("Selezione Landmark", mouse_callback)
            
            print("PRE: Clicca 2 punti anatomici evidenti. Poi premi INVIO/SPAZIO (o 'r' per ricliccare).")
            key = cv.waitKey(0) & 0xFF
            if key == ord('r'): continue
            elif len(clicked_points) == 2: break
        
        dist_px_pre = calculate_distance(clicked_points[0], clicked_points[1])
        real_distance_um = dist_px_pre * pre_scale
        
        # --- 3. CLICCA SULLA POST ---
        while True:
            clicked_points = []
            img_display = post_img.copy()
            cv.imshow("Selezione Landmark", img_display)
            cv.setMouseCallback("Selezione Landmark", mouse_callback)
            
            print("POST: Clicca gli STESSI 2 punti. Poi premi INVIO/SPAZIO (o 'r' per ricliccare).")
            key = cv.waitKey(0) & 0xFF
            if key == ord('r'): continue
            elif len(clicked_points) == 2: break
                
        dist_px_post = calculate_distance(clicked_points[0], clicked_points[1])
        
        # --- 4. CALCOLO E SALVATAGGIO ---
        post_scale = real_distance_um / dist_px_post
        print(f"Distanza reale: {real_distance_um:.1f} um | Distanza POST: {dist_px_post:.1f} px")
        print(f">>> SCALA POST-REGISTRAZIONE: {post_scale:.4f} um/px <<<")
        
        with open(CSV_OUTPUT_PATH, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([subject_name, round(pre_scale, 4), round(post_scale, 4)])
            
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()