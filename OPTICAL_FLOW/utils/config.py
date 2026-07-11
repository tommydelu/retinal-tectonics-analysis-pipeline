# Ogni coppia (min, max) viene eseguita come un ciclo completo separato su tutti i
# soggetti: min = soglia sotto la quale un vettore è troppo debole (probabile rumore
# statico), max = soglia sopra la quale un vettore è troppo forte (probabile outlier).
MOVEMENT_THRESHOLDS = [
    (1, 10),
]

PIXEL_LENGTH   = 4.651162790697675     # in um -- valore di default (dataset 1, risoluzione fissa)
INNER_DIAMETER_UM = 4000               # um --> 4 mm
OUTER_DIAMETER_UM = 8000               # um --> 8 mm

fovea_center_dict = {'L_06': (918, 1104), 'L_15': (1006, 987), 'L_26': (942, 937), 'L_30': (946, 946), 'L_42': (1016, 1008), 
 'L_48': (930, 958), 'L_63': (1078, 973), 'L_78': (949, 951), 'S_08': (977, 982), 'S_46': (920, 1123)}

PIXEL_LENGTH_MAPPING = {'1904,1912': [4.641411925308371, 4.660913656086978], '1398,1399': [3.407927453561504, 3.4103651699088293], 
                        '1906,1912': [4.646287358003023, 4.660913656086978], '1395,1330': [3.4006143045195265, 3.2421627419433476], 
                        '1417,1421': [3.4542440641606946, 3.463994929549998], '1674,1681': [4.0807371654234315, 4.097801179854712], 
                        '1676,1681': [4.085612598118083, 4.097801179854712], '1678,1678': [4.090488030812735, 4.090488030812735], 
                        '1398,1398': [3.407927453561504, 3.407927453561504], '1904,1911': [4.641411925308371, 4.658475939739652], 
                        '1489,1473': [3.629759641168154, 3.5907561796109406], '1414,1409': [3.4469309151187173, 3.434742333382088], 
                        '1357,1357': [3.307981083321145, 3.307981083321145], '1358,1358': [3.3104187996684704, 3.3104187996684704], 
                        '1363,1357': [3.3226073814050996, 3.307981083321145], '1415,1334': [3.449368631466043, 3.2519136073326513], 
                        '1536,1636': [3.7443323094924676, 3.9881039442250503], '1679,1679': [4.092925747160061, 4.092925747160061], 
                        '1363,1363': [3.3226073814050996, 3.3226073814050996], '1680,1680': [4.095363463507386, 4.095363463507386], 
                        '1492,1480': [3.6370727902101314, 3.607820194042221], '1427,1393': [3.478621227633953, 3.3957388718248747], 
                        '1449,1439': [3.5322509872751207, 3.5078738238018627], '943,1005': [2.2987665155282535, 2.4499049290624546], 
                        '1412,1429': [3.442055482424065, 3.4834966603286044], '1364,1364': [3.3250450977524255, 3.3250450977524255]}

# To generate PIXEL_LENGTH_MAPPING with meaningfull values
# pixel_length_list = []
# for key in PIXEL_LENGTH_MAPPING.keys():
#     list_key = key.split(",")
#     pixel_length_list.append( (int(list_key[0]) * PIXEL_LENGTH) / 1908 )
#     pixel_length_list.append( (int(list_key[1]) * PIXEL_LENGTH) / 1908 )
#     PIXEL_LENGTH_MAPPING[key] = pixel_length_list
#     pixel_length_list = []

# print(PIXEL_LENGTH_MAPPING)


def get_pixel_length(width: int, height: int) -> tuple[float, float]:
    """
    Restituisce (pixel_length_x, pixel_length_y) in um/pixel per la risoluzione data.

    Usa PIXEL_LENGTH_MAPPING se la risoluzione è presente (dataset 2, dove ogni immagine
    può avere una risoluzione diversa), altrimenti ricade su PIXEL_LENGTH per entrambi gli
    assi (dataset 1, risoluzione fissa e quadrata).
    """
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

