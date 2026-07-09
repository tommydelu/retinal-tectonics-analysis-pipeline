import os
import cv2
import gzip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from common.paths import PROJECT_ROOT

SRC_NIR_PATH1 = os.path.join(PROJECT_ROOT, 'DATA','DATASET1','raw','IR','S_36POST.JPG')
SRC_NIR_PATH2 = os.path.join(PROJECT_ROOT, 'DATA','DATASET1','raw','IR','S_52PRE.JPG')

SRC_STARE_PATH = os.path.join(PROJECT_ROOT,'PAPERS','myPaper','images_to_put_on_paper','im0001.ppm.gz')
SRC_DRIVE_PATH = os.path.join(PROJECT_ROOT,'PAPERS','myPaper','images_to_put_on_paper','21_training.tif')

DST_PATH = os.path.join(PROJECT_ROOT, 'PAPERS', 'myPaper', 'paper_imgs')


Im_nir_good = cv2.imread(SRC_NIR_PATH1) # 1908 x 1908
Im_nir_bad = cv2.imread(SRC_NIR_PATH2)

Im_drive = cv2.imread(SRC_DRIVE_PATH)
Im_drive = cv2.cvtColor(Im_drive, cv2.COLOR_BGR2RGB)

with gzip.open(SRC_STARE_PATH, "rb") as f:
        Im_stare = np.array(Image.open(f)) # 584 x 565

images = [Im_nir_good, Im_nir_bad, Im_drive, Im_stare]

for i,file in enumerate(images):
        
        plt.figure(figsize=(8,8))
        plt.imshow(file)
        plt.axis('off')
        plt.savefig(DST_PATH+f'/img{i}.pdf',dpi=300,format='pdf',bbox_inches='tight')
        plt.close()




