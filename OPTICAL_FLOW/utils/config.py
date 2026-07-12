# Ogni coppia (min, max) viene eseguita come un ciclo completo separato su tutti i
# soggetti: min = soglia sotto la quale un vettore è troppo debole (probabile rumore
# statico), max = soglia sopra la quale un vettore è troppo forte (probabile outlier).
MOVEMENT_THRESHOLDS = [
    (1, 10),
]

PIXEL_LENGTH   = 4.651162790697675     # in um -- valore di default (dataset 1, risoluzione fissa)
INNER_DIAMETER_UM = 4000               # um --> 4 mm
OUTER_DIAMETER_UM = 8000               # um --> 8 mm

# winsize di Farneback calibrato sulla risoluzione di riferimento (dataset 1, PIXEL_LENGTH).
FARNEBACK_BASE_WINSIZE = 31

fovea_center_dict = {'L_06': (918, 1104), 'L_15': (1006, 987), 'L_26': (942, 937), 'L_30': (946, 946), 'L_42': (1016, 1008), 
 'L_48': (930, 958), 'L_63': (1078, 973), 'L_78': (949, 951), 'S_08': (977, 982), 'S_46': (920, 1123)}

# Riferimento usato per generare PIXEL_LENGTH_MAPPING per proporzione: risoluzione (quadrata)
# alla quale è stato misurato PIXEL_LENGTH.
REFERENCE_RESOLUTION = 1908

PIXEL_LENGTH_MAPPING = {'1904,1912': [4.6609341410983, 4.641432324608349], '1398,1399': [6.347938916059488, 6.343401432917201],
                        '1906,1912': [4.656043339271335, 4.641432324608349], '1395,1330': [6.361590397599401, 6.672495191467041],
                        '1417,1421': [6.262821880487766, 6.245192543737624], '1674,1681': [5.301325331332834, 5.279249616092304],
                        '1676,1681': [5.2949991674529615, 5.279249616092304], '1678,1678': [5.288688083820718, 5.288688083820718],
                        '1398,1398': [6.347938916059488, 6.347938916059488], '1904,1911': [4.6609341410983, 4.643861122266438],
                        '1489,1473': [5.959985631061897, 6.024724103632834], '1414,1409': [6.276109338508602, 6.298380840774424],
                        '1357,1357': [6.539733680656716, 6.539733680656716], '1358,1358': [6.534917971024421, 6.534917971024421],
                        '1363,1357': [6.510945417939225, 6.539733680656716], '1415,1334': [6.271673925548526, 6.652487709633556],
                        '1536,1636': [5.777616279069768, 5.424461249786775], '1679,1679': [5.285538180256798, 5.285538180256798],
                        '1363,1363': [6.510945417939225, 6.510945417939225], '1680,1680': [5.282392026578074, 5.282392026578074],
                        '1492,1480': [5.948001745744748, 5.996228786926462], '1427,1393': [6.218933850491355, 6.370724052154461],
                        '1449,1439': [6.124512494583274, 6.167073387526869], '943,1005': [9.410836272164543, 8.830267268309616],
                        '1412,1429': [6.284999011792609, 6.210229954269534], '1364,1364': [6.5061719975448415, 6.5061719975448415]}

# Per generare PIXEL_LENGTH_MAPPING: si assume che le risoluzioni di dataset 2 rappresentino
# tutte (circa) lo stesso campo visivo anatomico, catturato con fotocamere di densità di pixel
# diversa — quindi pixel_length è inversamente proporzionale alla risoluzione (più pixel per
# la stessa scena reale --> ogni pixel copre meno spazio fisico), non direttamente
# proporzionale come in una prima versione di questa formula (quella dava anelli di raggio
# maggiore di mezza immagine per 23 risoluzioni su 26: implicava un campo visivo che si
# restringe per le immagini più piccole, il contrario di quanto atteso da un semplice
# ricampionamento della stessa scena).
# pixel_length_list = []
# for key in PIXEL_LENGTH_MAPPING.keys():
#     list_key = key.split(",")
#     pixel_length_list.append( (REFERENCE_RESOLUTION * PIXEL_LENGTH) / int(list_key[0]) )
#     pixel_length_list.append( (REFERENCE_RESOLUTION * PIXEL_LENGTH) / int(list_key[1]) )
#     PIXEL_LENGTH_MAPPING[key] = pixel_length_list
#     pixel_length_list = []

# print(PIXEL_LENGTH_MAPPING)

# Approssimazione (nessuna misura diretta della barra di scala per-immagine): resta
# un'assunzione, non una calibrazione vera come darebbe measure_pixel_um_dimension.py.
USE_PIXEL_LENGTH_MAPPING = True


def get_pixel_length(width: int, height: int) -> tuple[float, float]:
    """
    Restituisce (pixel_length_x, pixel_length_y) in um/pixel per la risoluzione data.

    Se USE_PIXEL_LENGTH_MAPPING è True e la risoluzione è presente in PIXEL_LENGTH_MAPPING
    (dataset 2, dove ogni immagine può avere una risoluzione diversa), usa quei valori;
    altrimenti ricade su PIXEL_LENGTH per entrambi gli assi (dataset 1, risoluzione fissa
    e quadrata, e qualunque risoluzione di dataset 2 non presente nella mappa).
    """
    if USE_PIXEL_LENGTH_MAPPING:
        key = f"{width},{height}"
        if key in PIXEL_LENGTH_MAPPING:
            pixel_length_x, pixel_length_y = PIXEL_LENGTH_MAPPING[key]
            return pixel_length_x, pixel_length_y
    return PIXEL_LENGTH, PIXEL_LENGTH


def compute_radii(pixel_length_x: float, pixel_length_y: float) -> tuple[int, int, int, int]:
    """
    Converte i diametri fisici (INNER/OUTER_DIAMETER_UM) in raggi in pixel, separatamente
    per asse x e y. Con pixel non quadrati (pixel_length_x != pixel_length_y) le zone
    anulari risultanti sono ellittiche in coordinate pixel, ma circolari in micrometri.
    """
    inner_radius_x = int(INNER_DIAMETER_UM / pixel_length_x / 2)
    inner_radius_y = int(INNER_DIAMETER_UM / pixel_length_y / 2)
    outer_radius_x = int(OUTER_DIAMETER_UM / pixel_length_x / 2)
    outer_radius_y = int(OUTER_DIAMETER_UM / pixel_length_y / 2)
    return inner_radius_x, inner_radius_y, outer_radius_x, outer_radius_y


def get_farneback_winsize(pixel_length_x: float, pixel_length_y: float) -> int:
    """
    Scala winsize (finestra di media di Farneback, in pixel) in proporzione alla
    pixel_length dell'immagine, così la finestra copre sempre la stessa area fisica
    (in um) indipendentemente dalla risoluzione: a pixel_length più piccola (immagine
    più fine) serve una finestra più larga in pixel per coprire la stessa area di
    riferimento calibrata su FARNEBACK_BASE_WINSIZE a risoluzione dataset 1.

    Arrotondato al dispari più vicino, come richiesto da cv.calcOpticalFlowFarneback.
    """
    pixel_length_avg = (pixel_length_x + pixel_length_y) / 2
    winsize = round(FARNEBACK_BASE_WINSIZE * PIXEL_LENGTH / pixel_length_avg)
    if winsize % 2 == 0:
        winsize += 1
    return max(winsize, 3)

