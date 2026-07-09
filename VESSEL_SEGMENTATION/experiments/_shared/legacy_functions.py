import cv2 as cv
import numpy as np


"""
First pre-processing attempt: use the pre-processing technique of [1] (read in the abstract)
"""

def pre_processing_attempt1(img, median_ksize = 3, clip_limit = 2, grid_size = (2,2), a_min = 0, a_max = 255):

    median_blur = cv.medianBlur(img, median_ksize) # blurring using the median of the pixels in a window of ksize x ksize

    """
    divido in quadratini di grandezza tileGridSize l'immagine, e migliora il contrasto separatamente per ogni quadratino
    esalto i contrasti, ma se il contrasto è troppo alto lo limito con clip limit
    """

    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    # con np.clip forzo tutti i valori dell'immagine a stare dentro il range a_min a_max
    clahe_img = np.clip(clahe.apply(median_blur), a_min, a_max) # per tenere l'img in un range controllato

    return clahe_img


#----------------------------------------------------------#


def extract_background(Io, linear_elements):

    min_opening = np.zeros_like(Io)+255

    for element in linear_elements:
        opening = cv.morphologyEx(Io, cv.MORPH_OPEN, element)
        min_opening = np.minimum(min_opening, opening)

    return min_opening


#----------------------------------------------------------#

def extract_background2(Io, linear_elements):

    max_opening = np.zeros_like(Io)

    for element in linear_elements:
        closing = cv.morphologyEx(Io, cv.MORPH_CLOSE, element)
        max_opening = np.maximum(max_opening, closing)

    return max_opening
