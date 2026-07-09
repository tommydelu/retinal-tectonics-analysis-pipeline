from pathlib import Path

# Root del progetto, calcolata dalla posizione di questo file: risolve sempre
# allo stesso percorso indipendentemente dalla working directory da cui
# viene lanciato uno script (a differenza di os.getcwd()).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
