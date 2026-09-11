import os
import re
import shutil
import matplotlib.pyplot as plt
from psd_tools import PSDImage

from common.paths import PROJECT_ROOT

# -------------------------------------------------------------
# CONFIGURAZIONE PERCORSI (Adatta GLOBAL_PATH alle tue esigenze)
# -------------------------------------------------------------
GLOBAL_PATH = PROJECT_ROOT
download_path = os.path.expanduser('~/Downloads/')

SRC_PATH = os.path.join(GLOBAL_PATH, 'DATA', 'DATASET2', 'raw', 'Immagini_IR')
os.makedirs(SRC_PATH, exist_ok=True)

psd_files = [os.path.join(download_path, f) for f in os.listdir(download_path) if f.endswith('.psd')]

MAPPA_MESI = {
    'preop': '0',
    '1_mese_postop': '1',
    '3_mesi_postop': '3',
    '12_mesi_postop': '12'
}

PATTERN_PREOP = re.compile(r'pre[- ]?op', re.IGNORECASE)
PATTERN_1MESE = re.compile(r'\b1\b\s*(mo|mese|mesi)', re.IGNORECASE)
PATTERN_3MESI = re.compile(r'\b3\b\s*(mo|mese|mesi)', re.IGNORECASE)
PATTERN_12MESI = re.compile(r'\b(12|30)\b\s*(mo|mese|mesi)', re.IGNORECASE)

# -------------------------------------------------------------
# FLAG DI DEBUG: se True, stampa e mostra tutti i livelli
# senza salvare nulla (utile per ispezionare il PSD)
# -------------------------------------------------------------
DEBUG_MODE = True


def sanifica_nome_file(nome):
    """Rende sicuro un nome livello da usare come nome file."""
    nome = nome.strip()
    nome = re.sub(r'[\\/*?:"<>|]', '_', nome)  # rimuove caratteri non validi su Windows/Mac
    nome = re.sub(r'\s+', '_', nome)           # spazi -> underscore
    return nome or "livello_senza_nome"


def debug_stampa_e_mostra_livelli(psd, nome_file, cartella_salvataggio):
    """Stampa a console tutti i livelli del PSD, li mostra a video
    e salva ciascuna immagine nella cartella di input."""
    print(f"\n--- Livelli trovati in '{nome_file}' ---")

    layers_da_mostrare = []

    for layer in psd.descendants():
        tipo = "GRUPPO" if layer.is_group() else "LIVELLO"
        print(f"  [{tipo}] id={layer.layer_id} | nome='{layer.name}' | "
              f"visibile={layer.visible} | size={layer.size}")

        if not layer.is_group():
            layers_da_mostrare.append(layer)

    print(f"--- Totale livelli non-gruppo: {len(layers_da_mostrare)} ---\n")

    if not layers_da_mostrare:
        return

    # --- Salvataggio di ogni livello nella cartella di input ---
    for layer in layers_da_mostrare:
        img = layer.topil()
        if img:
            nome_layer_pulito = sanifica_nome_file(layer.name)
            nome_out = f"{nome_file}__id{layer.layer_id}_{nome_layer_pulito}.png"
            percorso_out = os.path.join(cartella_salvataggio, nome_out)
            img.save(percorso_out)
            print(f"  -> Salvato livello '{layer.name}' in: {percorso_out}")

    # --- Griglia per mostrare tutte le immagini insieme ---
    n = len(layers_da_mostrare)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, layer in zip(axes, layers_da_mostrare):
        img = layer.topil()
        if img:
            ax.imshow(img)
        ax.set_title(layer.name, fontsize=9)
        ax.axis('off')

    for ax in axes[len(layers_da_mostrare):]:
        ax.axis('off')

    fig.suptitle(nome_file)
    plt.tight_layout()
    plt.show()


print("=== INIZIO ESTRAZIONE E RINOMINA INTEGRATA ===")

for psd_path in psd_files:
    if not os.path.exists(psd_path):
        continue

    nome_base = os.path.splitext(os.path.basename(psd_path))[0]
    nome_pulito = re.sub(r'(_def|_finito|_finito|_def)$', '', nome_base, flags=re.IGNORECASE)
    nome_pulito = nome_pulito.strip('_').strip()
    prefisso_ottico = f"{nome_pulito}IR_"

    print(f"\nElaborazione: {nome_base}.psd  -->  Prefisso export: {prefisso_ottico}")

    psd = PSDImage.open(psd_path)

    # --- MODALITA' DEBUG: stampa/mostra tutto e passa al file successivo ---
    if DEBUG_MODE:
        debug_stampa_e_mostra_livelli(psd, nome_base, download_path)
        continue

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

    if gruppi_livelli['preop']:
        lista_preop = gruppi_livelli['preop']
        if len(lista_preop) > 1:
            maiuscoli = [l for l in lista_preop if l.name == 'PREOP']
            livello_scelto = maiuscoli[0] if maiuscoli else lista_preop[0]
        else:
            livello_scelto = lista_preop[0]

        img = livello_scelto.topil()
        if img:
            nome_file_finale = f"{prefisso_ottico}{MAPPA_MESI['preop']}.png"
            img.save(os.path.join(SRC_PATH, nome_file_finale))
            print(f"  -> Salvato: {nome_file_finale}")

    for categoria in ['1_mese_postop', '3_mesi_postop', '12_mesi_postop']:
        lista_tuple = gruppi_livelli[categoria]
        if not lista_tuple:
            continue

        l_filtrati = [layer for layer, is_vaso in lista_tuple if not is_vaso]
        livelli_effettivi = l_filtrati if l_filtrati else [layer for layer, is_vaso in lista_tuple]

        livello_scelto = min(livelli_effettivi, key=lambda l: l.layer_id)

        img = livello_scelto.topil()
        if img:
            nome_file_finale = f"{prefisso_ottico}{MAPPA_MESI[categoria]}.png"
            img.save(os.path.join(SRC_PATH, nome_file_finale))
            print(f"  -> Salvato: {nome_file_finale}")

print("\n" + "="*50)
print(f"🎉 Operazione completata! I file rinominati sono pronti in:\n--> {SRC_PATH}")
print("="*50)