# Analisi di immagini retiniche

Pipeline di computer vision per l'analisi di immagini retiniche a infrarossi (NIR) pre/post intervento: segmentazione automatica dei vasi sanguigni, registrazione (allineamento) delle immagini PRE/POST, e stima dell'optical flow tra le due fasi.

## Struttura del repository

```
DATA/                  # dataset (dati sensibili, non versionati — vedi sezione Dati)
VESSEL_SEGMENTATION/    # segmentazione automatica dei vasi retinici
├── production/         # pipeline finale/attuale
├── experiments/        # iterazioni storiche che hanno portato alla pipeline finale
└── utils/               # calcolo metriche

IMAGE_REGISTRATION/     # allineamento PRE/POST delle immagini
└── src/

OPTICAL_FLOW/           # stima del movimento tra immagini PRE/POST
├── farneback/           # metodo Farneback (dataset 1 legacy / dataset 2 produzione)
├── lukas_kanade/         # metodo Lucas-Kanade (prototipo, incompleto)
├── segmentation/         # segmentazione applicata al pipeline optical flow
├── paper_scripts/        # script "usa e getta" per generare figure/tabelle del paper
└── utils/                # classi e funzioni condivise (ConfigPaths, Subject, config)

common/                 # libreria condivisa tra i tre domini sopra (filtri immagine, path)
scripts/                # script di supporto non legati a un singolo dominio
PAPERS/                 # materiale e script per la generazione delle figure del paper
```

## Installazione

Richiede Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dati

Gli script si aspettano i dati in:
- `DATA/DATASET1/raw/IR`, `DATA/DATASET1/raw/labels` — immagini e label del dataset 1
- `DATA/DATASET2/raw/...` — immagini del dataset 2

I dati contengono immagini cliniche di pazienti e **non sono versionati** (`DATA/DATASET1/raw`, `DATA/DATASET2/raw` e `OPTICAL_FLOW/results/ds2` sono in `.gitignore`). Per eseguire gli script serve popolare queste cartelle con i propri dati, mantenendo la struttura sopra.

Tutti i path sono calcolati a partire dalla root del progetto (`common/paths.py`), quindi gli script funzionano indipendentemente dalla cartella da cui vengono lanciati.

## Pipeline di segmentazione dei vasi (`VESSEL_SEGMENTATION/`)

- `production/` contiene la pipeline attuale: `explore_big_vessels.py` / `explore_thin_vessels.py` per tarare i parametri su una singola immagine, `batch_segment_big_vessels.py` / `batch_segment_thin_vessels.py` per elaborare l'intero dataset, `merge_vessel_masks.py` per unire le due maschere, `select_best_subjects.py` e `evaluate_predictions_vs_gt.py` per selezione e validazione post-hoc.
- `experiments/iteration_1`, `iteration_2`, `iteration_3` sono i tre tentativi metodologici precedenti, mantenuti per riferimento storico (ognuno rappresenta un passo dell'evoluzione dell'algoritmo finale).

## Pipeline di registrazione delle immagini (`IMAGE_REGISTRATION/src/`)

Ordine di esecuzione:
1. `pick_fovea_coordinates.py` — tool interattivo, click manuale sul centro della fovea
2. `locate_optic_nerve.py` — localizzazione automatica del nervo ottico (dataset 2)
3. `align_pre_post_images.py` — allineamento SIFT PRE vs POST/1mo/3mo/12mo (dataset 2)
4. `compute_registration_metrics.py` — calcolo RMSE contro un riferimento manuale

`legacy_dataset1_overlay.py` è un ramo indipendente più vecchio, per il dataset 1, mantenuto per riferimento.

## Pipeline di optical flow (`OPTICAL_FLOW/`)

- `farneback/src/`: pipeline principale, basata su classi. Entry point unico `run_farneback_optical_flow.py`:
  ```bash
  python -m OPTICAL_FLOW.farneback.src.run_farneback_optical_flow --dataset 1 --mode interpolated
  python -m OPTICAL_FLOW.farneback.src.run_farneback_optical_flow --dataset 2 --mode masked --save-figures
  ```
  - `--dataset {1,2}`: dataset 1 (10 soggetti fissi, confronto PRE/POST) o dataset 2 (tutti i soggetti disponibili, confronto baseline vs 12 mesi post-operatorio)
  - `--mode interpolated`: calcola l'optical flow sui pixel dei vasi e lo interpola (RBF, thin plate spline) su tutta l'immagine — campo denso, cache su `.npy` tra esecuzioni
  - `--mode masked`: nessuna interpolazione, le metriche si calcolano direttamente sul campo di Farneback grezzo, ristretto ai soli pixel dei vasi
  - Entrambe le modalità calcolano lo stesso schema di metriche (intera immagine, ROI totale, anello interno/esterno, 8 quadranti × X/Y/magnitude), salvate in `dataset{1,2}_{interpolated,masked}_metrics.csv`
  - Classi principali: `RetinalZoneMasks` (maschere geometriche), `RBFFlowInterpolator` (interpolazione), `FarnebackFlowField` (calcolo Farneback + maschera di validità), `Dataset1Subjects`/`Dataset2Subjects` (caricamento soggetti)
- `lukas_kanade/`: `optical_flow_lucas_kanade.py`, prototipo parziale (gran parte della pipeline è commentata, esegue solo il feature matching SIFT+FLANN)
- `segmentation/`: segmentazione dei vasi applicata alle immagini di input dell'optical flow

## Script di supporto

- `scripts/import_psd_images.py` — estrae le immagini dai file PSD del dataset 2
- `scripts/merge_optical_flow_csv.py` — unisce i CSV dei risultati dell'optical flow
- `PAPERS/myPaper/` — script e materiali per generare le figure del paper
- `OPTICAL_FLOW/paper_scripts/` — script per calibrazione scala px→µm e tabelle riassuntive

## Nota sul refactor della pipeline Farneback

Durante la riscrittura di `OPTICAL_FLOW/farneback/` è stato corretto (su richiesta esplicita) un bug per cui, per il dataset 1, la maschera dei vasi usata per validare l'optical flow veniva letta dall'output della segmentazione automatica invece che dalle label manuali/ground-truth (il path corretto veniva calcolato ma non era mai usato). I numeri prodotti oggi da `--dataset 1` differiscono quindi da quelli storici in `RBF_pred_1.csv`.

## Problemi noti (non corretti in questa riorganizzazione)

Durante il riordino sono stati individuati alcuni problemi preesistenti nel codice, lasciati intenzionalmente invariati per non alterare i risultati numerici:

- `VESSEL_SEGMENTATION/utils/compute_metrics.py` chiama `get_metrics(pred, label)` ma la firma della funzione è `get_metrics(label, pred)`: gli argomenti sono invertiti (Sensitivity/Specificity/False Positive Rate risultano scambiati tra loro; Dice Score, Jaccard Index e MCC non sono affetti per simmetria).
- `VESSEL_SEGMENTATION/utils/metrics_functions.py` contiene una funzione `accuracy()` con firma incompleta (mai chiamata, codice morto).
- `OPTICAL_FLOW/lukas_kanade/src/optical_flow_lucas_kanade.py` non ha mai una pipeline completa funzionante: referenzia un file `segmentation_1.csv` che non esiste nella struttura dati attuale (esiste solo `seg_metrics1.csv`), quindi fallisce se eseguito così com'è. Il problema era presente anche prima di questa riorganizzazione.
- La pipeline `production/` di `VESSEL_SEGMENTATION` ha suffissi numerici incoerenti tra i vari step (es. alcuni script producono `_5`, altri consumano `_2`), segno che la pipeline non è mai stata riallineata dopo l'ultima iterazione di sviluppo.
