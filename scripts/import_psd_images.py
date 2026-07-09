import os
import re
import shutil
from psd_tools import PSDImage

from common.paths import PROJECT_ROOT

# -------------------------------------------------------------
# CONFIGURAZIONE PERCORSI (Adatta GLOBAL_PATH alle tue esigenze)
# -------------------------------------------------------------
GLOBAL_PATH = PROJECT_ROOT
download_path = os.path.expanduser('~/Downloads/')

# Il path di destinazione richiesto per l'Optical Flow
SRC_PATH = os.path.join(GLOBAL_PATH, 'DATA', 'DATASET2', 'raw', 'Immagini_IR')
os.makedirs(SRC_PATH, exist_ok=True)

psd_files = [os.path.join(download_path, f) for f in os.listdir(download_path) if f.endswith('.psd')]

# Mappatura dei nomi dei file estratti con i suffissi numerici richiesti
MAPPA_MESI = {
    'preop': '0',
    '1_mese_postop': '1',
    '3_mesi_postop': '3',
    '12_mesi_postop': '12'
}

# -------------------------------------------------------------
# REGEX PER ESTRAZIONE (Invariate, super robuste)
# -------------------------------------------------------------
PATTERN_PREOP = re.compile(r'pre[- ]?op', re.IGNORECASE)
PATTERN_1MESE = re.compile(r'\b1\b\s*(mo|mese|mesi)', re.IGNORECASE)
PATTERN_3MESI = re.compile(r'\b3\b\s*(mo|mese|mesi)', re.IGNORECASE)
PATTERN_12MESI = re.compile(r'\b(12|30)\b\s*(mo|mese|mesi)', re.IGNORECASE)

print("=== INIZIO ESTRAZIONE E RINOMINA INTEGRATA ===")

for psd_path in psd_files:
    if not os.path.exists(psd_path):
        continue
        
    # 1. PULIZIA DEL NOME DEL SOGGETTO (CORRETTA)
    # Estrae il nome base (es. "Cognome_finito" o "Cognome_def")
    nome_base = os.path.splitext(os.path.basename(psd_path))[0]

    # Rimuove in modo sicuro i suffissi di stato alla fine del file, ignorando maiuscole/minuscole
    nome_pulito = re.sub(r'(_def|_finito|_finito|_def)$', '', nome_base, flags=re.IGNORECASE)

    # Rimuove eventuali spazi o underscore residui alla fine prima di attaccare IR
    nome_pulito = nome_pulito.strip('_').strip()

    # Crea ESATTAMENTE il prefisso richiesto attaccando "IR_" (es. "CognomeIR_")
    prefisso_ottico = f"{nome_pulito}IR_"
    
    print(f"\nElaborazione: {nome_base}.psd  -->  Prefisso export: {prefisso_ottico}")
    
    psd = PSDImage.open(psd_path)
    
    gruppi_livelli = {
        'preop': [],
        '1_mese_postop': [],
        '3_mesi_postop': [],
        '12_mesi_postop': []
    }
    
    for layer in psd.descendants():
        if layer.is_group():
            continue
            
        nome_min = layer.name.strip().lower()
        is_only_vaso = ('mo' in nome_min or 'mo ' in nome_min) and 'post' not in nome_min
        
        if PATTERN_PREOP.search(nome_min):
            gruppi_livelli['preop'].append(layer)
        elif PATTERN_1MESE.search(nome_min):
            gruppi_livelli['1_mese_postop'].append((layer, is_only_vaso))
        elif PATTERN_3MESI.search(nome_min):
            gruppi_livelli['3_mesi_postop'].append((layer, is_only_vaso))
        elif PATTERN_12MESI.search(nome_min):
            gruppi_livelli['12_mesi_postop'].append((layer, is_only_vaso))

    # 2. SALVATAGGIO DIRETTO IN VERSIONE RINOMINATA DENTRO "Immagini_IR"
    # Gestione singola del PREOP
    if gruppi_livelli['preop']:
        lista_preop = gruppi_livelli['preop']
        if len(lista_preop) > 1:
            maiuscoli = [l for l in lista_preop if l.name == 'PREOP']
            livello_scelto = maiuscoli[0] if maiuscoli else lista_preop[0]
        else:
            livello_scelto = lista_preop[0]
            
        img = livello_scelto.topil()
        if img:
            # Nome formato: CognomeIR_0.png
            nome_file_finale = f"{prefisso_ottico}{MAPPA_MESI['preop']}.png"
            img.save(os.path.join(SRC_PATH, nome_file_finale))
            print(f"  -> Salvato: {nome_file_finale}")

    # Gestione delle altre fasi post-op
    for categoria in ['1_mese_postop', '3_mesi_postop', '12_mesi_postop']:
        lista_tuple = gruppi_livelli[categoria]
        if not lista_tuple:
            continue
            
        l_filtrati = [layer for layer, is_vaso in lista_tuple if not is_vaso]
        livelli_effettivi = l_filtrati if l_filtrati else [layer for layer, is_vaso in lista_tuple]
        
        livello_scelto = min(livelli_effettivi, key=lambda l: l.layer_id)
        
        img = livello_scelto.topil()
        if img:
            # Es: CognomeIR_1.png, CognomeIR_3.png, CognomeIR_12.png
            nome_file_finale = f"{prefisso_ottico}{MAPPA_MESI[categoria]}.png"
            img.save(os.path.join(SRC_PATH, nome_file_finale))
            print(f"  -> Salvato: {nome_file_finale}")

print("\n" + "="*50)
print(f"🎉 Operazione completata! I file rinominati sono pronti in:\n--> {SRC_PATH}")
print("="*50)