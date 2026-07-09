import os
import pandas as pd

from common.paths import PROJECT_ROOT

# -------------------------------------------------------------
# PERCORSI (Gli stessi del tuo script principale)
# -------------------------------------------------------------
DST_BASE_PATH = os.path.join(PROJECT_ROOT, 'OPTICAL_FLOW', 'results', 'ds2', 'farneback', 'results')
CSV_RESULTS_PATH = os.path.join(DST_BASE_PATH, 'data', 'CSVs')

# I nomi dei file da unire
file_vecchio = os.path.join(CSV_RESULTS_PATH, 'all_metrics_results.csv')
file_nuovo = os.path.join(CSV_RESULTS_PATH, 'ALL_METRICS_RESULTS_MASTER.csv')

# Il nome del file finale che conterrà tutti i 32 soggetti
file_finale = os.path.join(CSV_RESULTS_PATH, 'DATASET2_COMPLETE_METRICS.csv')

# -------------------------------------------------------------
# LOGICA DI UNIONE
# -------------------------------------------------------------
print("🔄 Avvio fusione dei file CSV...")

if os.path.exists(file_vecchio) and os.path.exists(file_nuovo):
    # Caricamento
    df_vecchio = pd.read_csv(file_vecchio)
    df_nuovo = pd.read_csv(file_nuovo)
    
    # Unione vera e propria (concatena le righe e resetta gli indici)
    df_completo = pd.concat([df_vecchio, df_nuovo], ignore_index=True)
    
    # Salvataggio
    df_completo.to_csv(file_finale, index=False)
    
    print("\n" + "="*50)
    print("✅ UNIONE COMPLETATA CON SUCCESSO!")
    print("="*50)
    print(f"Righe file vecchio (11 soggetti): {len(df_vecchio)}")
    print(f"Righe file nuovo   (21 soggetti): {len(df_nuovo)}")
    print(f"Righe file FINALE  (32 soggetti): {len(df_completo)}")
    print("="*50)
    print(f"Troverai il file completo qui:\n👉 {file_finale}\n")
    
else:
    print("\n❌ ERRORE: Impossibile trovare uno o entrambi i file nella cartella CSV.")
    if not os.path.exists(file_vecchio): print(f"   Manca: {file_vecchio}")
    if not os.path.exists(file_nuovo): print(f"   Manca: {file_nuovo}")