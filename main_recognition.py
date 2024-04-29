import os
import re
import cv2
import easyocr

import pandas as pd
from PIL import Image
from cv2 import detail_ExposureCompensator
from os.path import dirname, join, abspath

from pipeline import detection
from crop_image_words import imageCrop

reader = easyocr.Reader(['en'], gpu=False)
base_path = dirname(abspath(__file__))
result_path = join(base_path, 'result-detection')

def recognition(list_images):
    final_result = []
    for images in list_images:
        result = reader.recognize(images, paragraph=True, decoder='greedy', allowlist='1234567890', detail=0)
        final_result.append(result[0])
        
    return final_result