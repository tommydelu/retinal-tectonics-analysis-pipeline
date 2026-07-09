"""
Estrae dal CSV delle metriche di segmentazione (seg_metrics1.csv) l'elenco dei soggetti
del dataset 1 per cui l'immagine IR risulta flippata rispetto ai vasi annotati (label):
la colonna 'Flipped' viene decisa in VESSEL_SEGMENTATION/utils/compute_metrics.py
scegliendo, tra label originale e label specchiata, quella con Dice Score più alto
contro la segmentazione automatica.

Il risultato viene scritto in un file .txt (un soggetto per riga) usato da
Dataset1Subjects per sapere quali maschere vanno flippate prima del calcolo dell'optical
flow, senza dover rileggere il CSV ad ogni run.

Uso:
    python -m OPTICAL_FLOW.segmentation.src.utils.extract_flipped_subjects
"""

import os
import pandas as pd

from common.paths import PROJECT_ROOT

CSV_SEG_PATH = os.path.join(PROJECT_ROOT, 'OPTICAL_FLOW', 'segmentation', 'results', 'data', 'seg_metrics1.csv')
OUT_PATH = os.path.join(PROJECT_ROOT, 'vasi_specchiati_dataset1.txt')


def extract_flipped_subjects(csv_path: str = CSV_SEG_PATH, out_path: str = OUT_PATH) -> list[str]:
    df = pd.read_csv(csv_path, index_col=False, usecols=['SUBJECT', 'Flipped'])
    df = df[df['SUBJECT'] != 'METRIC MEAN'].copy()
    df['subject'] = df['SUBJECT'].str.replace('PRE', '', regex=False).str.replace('POST', '', regex=False)
    df['phase'] = df['SUBJECT'].apply(lambda s: 'PRE' if 'PRE' in s else 'POST')

    pivot = df.pivot(index='subject', columns='phase', values='Flipped')
    mismatches = pivot[pivot['PRE'] != pivot['POST']]
    for subject, row in mismatches.iterrows():
        print(f"[!] {subject}: PRE e POST discordanti (PRE={row['PRE']}, POST={row['POST']}) "
              f"— uso il valore di PRE, ma controllalo a mano")

    flipped_subjects = sorted(pivot[pivot['PRE'] == True].index.tolist())

    with open(out_path, 'w') as f:
        f.write('\n'.join(flipped_subjects))

    print(f"Scritti {len(flipped_subjects)} soggetti in {out_path}")
    return flipped_subjects


if __name__ == "__main__":
    extract_flipped_subjects()
