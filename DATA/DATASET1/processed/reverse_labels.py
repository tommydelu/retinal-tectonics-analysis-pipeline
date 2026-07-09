"""
Since i performed the segmentation having bright vessels on a dark background this code is used to reverse the labels
"""

import os
import cv2 as cv

from common.paths import PROJECT_ROOT

LABELS_PATH = os.path.join(PROJECT_ROOT,'DATA','DATASET1','raw','labels')
DST_PATH = os.path.join(PROJECT_ROOT,'DATA','DATASET1','processed','labels_reversed')

os.makedirs(DST_PATH, exist_ok=True)


if __name__ == '__main__':

    for dir in sorted(os.listdir(LABELS_PATH)):

        SUBJECT_PATH = os.path.join(LABELS_PATH,dir)

        output_path1 = os.path.join(DST_PATH,dir)
        output_path2 = os.path.join(DST_PATH,dir)

        os.makedirs(output_path1, exist_ok=True)
        os.makedirs(output_path2, exist_ok=True)

        img1_path = os.path.join(SUBJECT_PATH,'total_1.png')
        img2_path = os.path.join(SUBJECT_PATH,'total_2.png')

        img1 = cv.imread(img1_path,0)
        img2 = cv.imread(img2_path,0)

        img1 = 255 - img1
        img2 = 255 - img2

        new_img1_path = os.path.join(output_path1,'total_1.png')
        new_img2_path = os.path.join(output_path1,'total_2.png')

        cv.imwrite(new_img1_path,img1)
        cv.imwrite(new_img2_path,img2)






















    



