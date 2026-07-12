import cv2 as cv
import numpy as np
import os
import pandas as pd

from common.image_filters import clahe


def create_subject_data(fname: str,best_subjects: list, src_path, masks_path, labels_path, csv_seg_path,draw_src_path,draw_option) -> dict:

    subject_dict = dict()

    if 'POST' in fname: # ad ogni iterazione processo la coppia PRE-POST
         return None

    subject = fname.split('PRE')[0] # esempio: da L_06PRE.JPG estraggo L_06
    phase = 'PRE'
    subject_phase = subject + phase

    # Se il soggetto non è tra i best non ci interessa
    if subject not in best_subjects:
        return None

    fname_post = f"{subject}POST.JPG" # estraggo il nome del file POST


    IMG_PATH1 = os.path.join(src_path,fname)
    IMG_PATH2 = os.path.join(src_path,fname_post)

    MASK_PATH1 = os.path.join(masks_path,fname)
    MASK_PATH2 = os.path.join(masks_path,fname_post)

    LABEL_PATH1 = os.path.join(labels_path,subject,'total_1.png')
    LABEL_PATH2 = os.path.join(labels_path,subject,'total_2.png')

    imgPre  = cv.imread(IMG_PATH1,0)
    imgPre = clahe(imgPre,2,4)

    imgPost = cv.imread(IMG_PATH2,0)
    imgPost = clahe(imgPost,2,4)

    maskPre  = cv.imread(MASK_PATH1,0)
    maskPost = cv.imread(MASK_PATH2,0)

    labelPre  = cv.imread(LABEL_PATH1,0)
    labelPost = cv.imread(LABEL_PATH2,0)

    # Controllo se la label è flippata in modo giusto o no
    flipped_cols_csv = pd.read_csv(csv_seg_path, index_col=False,usecols=['SUBJECT','Flipped'])
    flipped_subject_state = flipped_cols_csv.loc[flipped_cols_csv['SUBJECT'] == subject_phase,'Flipped'].values[0]

    if flipped_subject_state: # se la label è specchiata rispetto all'immagine
        maskPre = cv.flip(maskPre,1)
        maskPost = cv.flip(maskPost,1)
        labelPre = cv.flip(labelPre,1)
        labelPost = cv.flip(labelPost,1)


    if draw_option == 3:
        fname_colored_superposition = os.path.join(draw_src_path,f'{subject}_confronto_post.png')
        img_colors = cv.imread(fname_colored_superposition)
        img_colors = cv.cvtColor(img_colors,cv.COLOR_BGR2RGB)

    elif draw_option == 1:
        img_colors = cv.cvtColor(imgPre,cv.COLOR_GRAY2BGR)
    else:
        img_colors = cv.cvtColor(labelPre,cv.COLOR_GRAY2BGR)

    subject_dict['ID'] = subject
    subject_dict['PHASE'] = phase
    subject_dict['ID and PHASE'] = subject_phase
    subject_dict['IMG PRE'] = imgPre
    subject_dict['IMG POST'] = imgPost
    subject_dict['LABEL PRE'] = labelPre
    subject_dict['LABEL POST'] = labelPost
    subject_dict['DRAW IMG'] = img_colors

    return subject_dict



def safe_mean(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Se non c'è nessun elemento di mask = true su cui calcolare la media, ritorna NaN
    # (dato mancante) invece di 0, per non confonderlo con una media realmente nulla.
    return np.mean(data[mask]) if np.any(mask) else np.nan
