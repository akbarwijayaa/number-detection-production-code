import os
import cv2
import numpy as np
import pandas as pd
from os.path import dirname, abspath, join

class imageCrop:
    # Crop image result pipeline
    
    def __init__(self, df_temp, img_nparr):
        self.df_temp = df_temp
        self.image = img_nparr
        self.score_bbox = str(self.df_temp['word_bboxes'][0]).split('),')
        self.bbox_coordinate = self.generate_words(self.score_bbox)
        self.sorted_bboxes = sorted(self.bbox_coordinate, key=lambda bbox: bbox[0])
            

    def crop(self, pts, img_nparr):
        """
        Takes inputs as 8 points
        and Returns cropped, masked image with a white background
        """
        rect = cv2.boundingRect(pts)
        x,y,w,h = rect
        cropped = img_nparr[y:y+h, x:x+w].copy()
        pts = pts - pts.min(axis=0)
        mask = np.zeros(cropped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        dst = cv2.bitwise_and(cropped, cropped, mask=mask)
        bg = np.ones_like(cropped, np.uint8)*255
        cv2.bitwise_not(bg,bg, mask=mask)
        dst2 = bg + dst
        return dst2
    
    def generate_words(self, score_bbox):
        num_bboxes = len(score_bbox)
        bbox = []
        for num in range(num_bboxes):
            bbox_coords = score_bbox[num].split(':')[-1].split(',\n')
            if bbox_coords!=['{}']:
                l_t = float(bbox_coords[0].strip(' array([').strip(']').split(',')[0]) # x1
                t_l = float(bbox_coords[0].strip(' array([').strip(']').split(',')[1]) # y1
                r_t = float(bbox_coords[1].strip(' [').strip(']').split(',')[0]) # param W / x2
                t_r = float(bbox_coords[1].strip(' [').strip(']').split(',')[1]) 
                r_b = float(bbox_coords[2].strip(' [').strip(']').split(',')[0])
                b_r = float(bbox_coords[2].strip(' [').strip(']').split(',')[1])
                l_b = float(bbox_coords[3].strip(' [').strip(']').split(',')[0])
                b_l = float(bbox_coords[3].strip(' [').strip(']').split(',')[1].strip(']')) # param h
                pts = np.array([[int(l_t), int(t_l)], [int(r_t) ,int(t_r)], [int(r_b) , int(b_r)], [int(l_b), int(b_l)]])
                
                # Parameter x y w h
                x = int(l_t)
                y = int(t_l)
                w = int(r_t - l_t)
                h = int(b_l - t_l)

                aaa = [x, y, w, h]
                bbox.append(aaa)
        return bbox

    def get_images(self):
        num_bboxes = len(self.score_bbox)
        a = 1
        result_images = []
        bbox_coor = []
        for x,y,w,h in self.sorted_bboxes:
            list_a = [x,y,w,h]
            for num in range(num_bboxes):
                bbox_coords = self.score_bbox[num].split(':')[-1].split(',\n')
                if bbox_coords!=['{}']:
                    l_t = float(bbox_coords[0].strip(' array([').strip(']').split(',')[0]) # x1
                    t_l = float(bbox_coords[0].strip(' array([').strip(']').split(',')[1]) # y1
                    r_t = float(bbox_coords[1].strip(' [').strip(']').split(',')[0]) # param W / x2
                    t_r = float(bbox_coords[1].strip(' [').strip(']').split(',')[1]) 
                    r_b = float(bbox_coords[2].strip(' [').strip(']').split(',')[0])
                    b_r = float(bbox_coords[2].strip(' [').strip(']').split(',')[1])
                    l_b = float(bbox_coords[3].strip(' [').strip(']').split(',')[0])
                    b_l = float(bbox_coords[3].strip(' [').strip(']').split(',')[1].strip(']')) # param h
                    pts = np.array([[int(l_t), int(t_l)], [int(r_t) ,int(t_r)], [int(r_b) , int(b_r)], [int(l_b), int(b_l)]])
                x1 = int(l_t)
                y1 = int(t_l)
                w1 = int(r_t - l_t)
                h1 = int(b_l - t_l)
                list_b = [x1,y1,w1,h1]
                word = self.crop(pts, self.image)       
                
                if list_a == list_b:
                    try:
                        result_images.append(word)
                        bbox_coor.append(list_b)
                        a = a+1
                    except:
                        continue 
        return result_images, bbox_coor