import os
import cv2 as cv

from common.paths import PROJECT_ROOT

ds1_path = os.path.join(PROJECT_ROOT, "DATA", "DATASET1",'raw','IR')
ds2_path = os.path.join(PROJECT_ROOT, "DATA", "DATASET2",'raw','Immagini_IR')

resolutions = set()

for f in os.listdir(ds2_path):
    img_path = os.path.join(ds2_path, f)
    img = cv.imread(img_path)
    h, w = img.shape[:2]
    resolutions.add((w, h))

print(f"Resolutions in DATASET1: {resolutions}")


