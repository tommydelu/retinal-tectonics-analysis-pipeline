import os
import cv2 as cv
import numpy as np
from common.paths import PROJECT_ROOT

# --- PATH DELLE CARTELLE ---
DS2_DIR_PATH1 = os.path.join(PROJECT_ROOT, 'DATA', 'DATASET2', 'raw', 'Immagini_IR')
DS2_DIR_PATH2 = os.path.join(PROJECT_ROOT, 'DATA', 'DATASET2', 'raw', 'IR_Def2')
DS2_DIR_PATH3 = os.path.join(PROJECT_ROOT, 'DATA', 'DATASET2', 'raw', 'IR_Def')

OUT_DIR = os.path.join(PROJECT_ROOT, 'DATA', 'DATASET2', 'raw', 'Immagini_IR_Corrette')
os.makedirs(OUT_DIR, exist_ok=True)

def get_all_pre_images(subject_name_post):
    """
    Trova TUTTE le immagini pre-registrazione del paziente in modo sicuro.
    Se post è 'AlbaneseAIR', la radice diventa 'albanesea'.
    """
    # Rimuoviamo SOLO 'IR' se si trova alla fine della stringa
    if subject_name_post.endswith('IR'):
        base_name = subject_name_post[:-2]
    else:
        base_name = subject_name_post
        
    base_name_lower = base_name.lower()
    
    pre_list = []
    for folder in [DS2_DIR_PATH2, DS2_DIR_PATH3]:
        if not os.path.exists(folder): continue
        for root, dirs, files in os.walk(folder):
            # IL SORTED QUI E' FONDAMENTALE
            for fname in sorted(files):
                if fname == '.DS_Store':
                    continue
                # Controlliamo se il file inizia con il nome del soggetto
                if fname.lower().startswith(base_name_lower):
                    pre_list.append(os.path.join(root, fname))
                    
    return pre_list

def get_all_post_images(subject_name):
    """Trova TUTTE le immagini post-registrazione di un paziente."""
    post_list = []
    # SORTED ANCHE QUI SULLE POST
    for fname in sorted(os.listdir(DS2_DIR_PATH1)):
        if fname == '.DS_Store':
            continue
        if fname.startswith(subject_name):
            post_list.append(os.path.join(DS2_DIR_PATH1, fname))
            
    # Assicuriamoci che la baseline (_0) sia il primo elemento
    post_list.sort(key=lambda x: "0" not in os.path.basename(x)) 
    return post_list

def resize_for_display(img, target_height=700):
    h, w = img.shape[:2]
    aspect_ratio = w / h
    new_w = int(target_height * aspect_ratio)
    return cv.resize(img, (new_w, target_height))

def draw_text(img, text):
    cv.putText(img, text, (15, 35), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv.putText(img, text, (15, 35), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return img

def main():
    print(f"Salvataggio in: {OUT_DIR}\n")
    print("=== COMANDI TASTIERA ===")
    print(" 'z'     -> Cambia immagine PRE (lato Sinistro)")
    print(" 'x'     -> Cambia immagine POST (lato Destro)")
    print(" 'h'     -> Specchia Orizzontale (Destra/Sinistra)")
    print(" 'v'     -> Specchia Verticale (Alto/Basso)")
    print(" 'r'     -> Resetta specchiature")
    print(" 'INVIO' -> Salva la baseline POST (_0) specchiata e passa al prossimo")
    print(" 'q'     -> Esci\n")

    post_files = sorted(os.listdir(DS2_DIR_PATH1))
    post_baselines = [f for f in post_files if f.endswith('_0.png') or f.endswith('_0.JPG')]

    for post_fname in post_baselines:
        # Estraiamo il nome esatto (es. "AlbaneseAIR")
        subject_name = post_fname.replace('_0.png', '').replace('_0.JPG', '')
        
        pre_paths = get_all_pre_images(subject_name)
        post_paths = get_all_post_images(subject_name)
        
        if not pre_paths:
            print(f"[{subject_name}] ATTENZIONE: Nessuna immagine PRE trovata. Copio la POST originale...")
            baseline_post_img = cv.imread(os.path.join(DS2_DIR_PATH1, post_fname))
            cv.imwrite(os.path.join(OUT_DIR, post_fname), baseline_post_img)
            continue
            
        pre_idx = 0
        post_idx = 0
        flip_h = False
        flip_v = False
        
        while True:
            current_pre_path = pre_paths[pre_idx]
            current_post_path = post_paths[post_idx]
            
            pre_img = cv.imread(current_pre_path)
            post_img = cv.imread(current_post_path)
            
            if flip_h: post_img = cv.flip(post_img, 1)
            if flip_v: post_img = cv.flip(post_img, 0)
            
            disp_pre = resize_for_display(pre_img)
            disp_post = resize_for_display(post_img)
            
            pre_name = os.path.basename(current_pre_path)
            post_name = os.path.basename(current_post_path)
            
            flip_text = ""
            if flip_h: flip_text += " [Flip H]"
            if flip_v: flip_text += " [Flip V]"
            
            draw_text(disp_pre, f"PRE: {pre_name}")
            draw_text(disp_post, f"POST: {post_name}{flip_text}")
            
            side_by_side = np.concatenate((disp_pre, disp_post), axis=1)
            window_name = f"Allineamento: {subject_name}"
            cv.imshow(window_name, side_by_side)
            
            key = cv.waitKey(0) & 0xFF
            
            if key == ord('z'):
                pre_idx = (pre_idx + 1) % len(pre_paths)
            elif key == ord('x'):
                post_idx = (post_idx + 1) % len(post_paths)
            elif key == ord('h'):
                flip_h = not flip_h
            elif key == ord('v'):
                flip_v = not flip_v
            elif key == ord('r'):
                flip_h = False
                flip_v = False
            elif key == 13 or key == 32: # INVIO o SPAZIO
                # Salviamo la BASELINE (_0) applicando le trasformazioni scelte
                baseline_post_path = post_paths[0] 
                final_img = cv.imread(baseline_post_path)
                
                if flip_h: final_img = cv.flip(final_img, 1)
                if flip_v: final_img = cv.flip(final_img, 0)
                
                out_path = os.path.join(OUT_DIR, os.path.basename(baseline_post_path))
                cv.imwrite(out_path, final_img)
                print(f"[{subject_name}] Salvato '{os.path.basename(baseline_post_path)}' con {flip_text}")
                cv.destroyWindow(window_name)
                break
            elif key == ord('q') or key == 27:
                print("Uscita forzata.")
                cv.destroyAllWindows()
                return

    print("\nFinito! Tutte le baseline POST sono state corrette e salvate.")

if __name__ == "__main__":
    main()