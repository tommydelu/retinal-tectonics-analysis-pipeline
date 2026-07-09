import os
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from common.paths import PROJECT_ROOT

# --- MODIFICA QUESTI PATH CON I TUOI ---
# Inserisci il path esatto dell'immagine originale PRE del soggetto L_03
IMG_PATH_PRE = os.path.join(PROJECT_ROOT, "DATA", "DATASET1", "raw", "IR", "L_03PRE.JPG")
DST_PATH_PAPER = os.path.join(PROJECT_ROOT, "paper_single_originals")  # Cartella per le immagini singole del paper

os.makedirs(DST_PATH_PAPER, exist_ok=True)

# --- COSTANTI PER LA SCALE BAR ---
PIXEL_LENGTH = 4.651162790697675     
SCALE_BAR_UM = 500                   
SCALE_BAR_PX = SCALE_BAR_UM / PIXEL_LENGTH

if __name__ == "__main__":
    
    # Carica l'immagine originale in scala di grigi
    imgPre = cv.imread(IMG_PATH_PRE, 0)
    
    if imgPre is not None:
        h, w = imgPre.shape

        # --- CREAZIONE FIGURA SINGOLA (PAPER FORMAT) ---
        fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)

        # Mostra l'immagine originale
        ax.imshow(imgPre, cmap='gray')
        ax.axis('off')

        # Scale Bar in basso a destra
        margin_x, margin_y, bar_height = 30, 30, 6
        start_x = w - SCALE_BAR_PX - margin_x
        start_y = h - margin_y - bar_height

        # Salvataggio stretto
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        out_path_paper = os.path.join(DST_PATH_PAPER, "L_03_Original_NIR_paper.png")
        
        plt.savefig(out_path_paper, dpi=300, bbox_inches='tight', pad_inches=0.01)
        plt.close(fig)
        
        print(f" [+] Immagine originale L_03 salvata per il paper!")
    else:
        print("[-] Errore: Immagine non trovata. Controlla il path.")