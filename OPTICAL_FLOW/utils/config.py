import os
import pandas as pd
from common.paths import PROJECT_ROOT

#-----------------------------------------#
# Default values used for the Dataset 1.  #
#-----------------------------------------#
DEFAULT_FARNEBACK_WINSIZE = 31
DEFAULT_UM_PIXEL_LENGTH   = 4.651162790697675
DEFAULT_RESOLUTION = 1908


#-----------------------------------------#
# Fixed quantities to use to draw the two #
# circles                                 #
#-----------------------------------------#
INNER_UM_DIAMETER = 4000
OUTER_UM_DIAMETER = 8000

#-----------------------------------------------------------------------------------------------------------------------#
# Dictionary containing the fovea centers of the best ten subjects of Dataset 1 according to the automatic segmentation #
#-----------------------------------------------------------------------------------------------------------------------#
fovea_pos = {'L_06': (918, 1104), 'L_15': (1006, 987), 'L_26': (942, 937), 'L_30': (946, 946), 'L_42': (1016, 1008), 
                     'L_48': (930, 958), 'L_63': (1078, 973), 'L_78': (949, 951), 'S_08': (977, 982), 'S_46': (920, 1123)}


pixel_length_mapping_csv = os.path.join(PROJECT_ROOT, 'files', 'ds2_scale_bar_length.csv')
df_scales = pd.read_csv(pixel_length_mapping_csv) if os.path.exists(pixel_length_mapping_csv) else None

#--------------------------------#
# Given a subject name, extract  #
# the corresponding scale bar    #
# value from the csv map.        #
#--------------------------------#
def getPixelLength(subject: str) -> float:
    if df_scales is None:
        raise FileNotFoundError(f"File non trovato: {pixel_length_mapping_csv}")
    matches = df_scales.loc[df_scales['Soggetto'] == subject, 'Scala_POST_um_px'].values
    if len(matches) == 0:
        raise ValueError(f"Soggetto '{subject}' non trovato nel file di mappatura scale.")
    return float(matches[0])

#--------------------------------#
# Given a um_pixel_length this   #
# function return the pixel      #
# length of the diamaters.       #
#--------------------------------#
def computeRadii(pixel_length) -> tuple[int, int]:
    inner_radius = int(round((INNER_UM_DIAMETER / pixel_length) / 2))
    outer_radius = int(round((OUTER_UM_DIAMETER / pixel_length) / 2))
    return inner_radius, outer_radius

#--------------------------------#
# Given a um_pixel_length this   #
# function compute the correct   #
# Farneback winsize              #
#--------------------------------#
def getFarnebackWinsize(pixel_length: float) -> int:
    raw_winsize = round(DEFAULT_FARNEBACK_WINSIZE * DEFAULT_UM_PIXEL_LENGTH / pixel_length)    
    winsize = max(int(raw_winsize), 9)
    if winsize % 2 == 0:
        winsize += 1
    return winsize


















