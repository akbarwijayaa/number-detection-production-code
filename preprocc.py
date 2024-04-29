import os
import cv2
import numpy as np
from PIL import Image

def applyHistEqualized(img_ndarray):
    gray = cv2.cvtColor(img_ndarray, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    togray = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    return togray


class preprocessing():
    
    def __init__(self, path):
        self.skewCorr(path)
        self.normalizeThinning(path)
        self.scaling(path)
    
    def skewCorr(self, path):
        for image in os.listdir(path):
            img = cv2.imread(path+image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.bitwise_not(gray)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            coords = np.column_stack(np.where(thresh > 0))
            angle = cv2.minAreaRect(coords)[-1]
            # print('angle sebelum update: ', image+str(angle))
            if angle < -45:
                angle = -(90 + angle)
            elif angle > 45:
                angle = -(270 + angle)
            elif angle == 90:
                angle = 0 
            else:
                angle = -angle
            # print('angle setelah update: ', image+str(angle))
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h), 
                                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            cv2.imwrite(path+image, rotated)
        
    def normalizeThinning(self, path):
        for image in os.listdir(path):
            img = cv2.imread(path + image)
            norm_img = np.zeros((img.shape[0], img.shape[1]))
            img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
            # kernel = np.ones((5,5),np.uint8)
            # erosion = cv2.erode(img, kernel, iterations = 1)
            cv2.imwrite(path + image, img)
        
    def scaling(self, path):
        for image in os.listdir(path):
            img = Image.open(path + image)
            length_x, width_y = img.size 
            size = int(min(1, float(1024.0 / length_x)) * length_x), int(min(1, float(1024.0 / length_x)) * width_y)
            img.resize(size, Image.ANTIALIAS).save(path + image, dpi=(300, 300))
            