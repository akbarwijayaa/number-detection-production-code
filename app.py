import cv2
import os
import json
import argparse
import pandas as pd
import numpy as np
import base64
from os.path import dirname, join, abspath
from preprocc import applyHistEqualized
from flask import Flask, request

from pipeline import detection
from crop_image_words import imageCrop
# from main_recognition import recognition
from paddlerec import recognize

app = Flask(__name__)
app.config["DEBUG"] = True

DETECTION_URL = "/number-detection-prodcode/predict"


@app.route(DETECTION_URL, methods=["POST", "GET"])
def predict():
    if not request.method == "POST":
        return "<h1>Number detection production code</h1>"

    image_file = request.files.getlist("image")
    for img in image_file:
        image_bytes = img.read()
        # convert to numpy.ndarray
        img_b64 = base64.b64encode(image_bytes)
        nparr = np.frombuffer(base64.b64decode(img_b64), np.uint8)
        img_ndarray = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # get shape
        img_width, img_height = get_shape(img_ndarray)
        img_preprocc = applyHistEqualized(img_ndarray)
        # process
        data_csv, img_arr = detection(img_preprocc)
        crop_img = imageCrop(data_csv, img_arr)
        result_images, bbox = crop_img.get_images()
        recog_result = recognize(result_images)
        # recog_result = recognition(result_images)
        df = mkdf(bbox, recog_result)
        dict_df = df.to_dict(orient='records')
        result = {
            'data':dict_df,
            'img_width':img_width,
            'img_height':img_height
        }
        return result

def mkdf(bbox, recog_result):
    df = pd.DataFrame(bbox, columns=['bbox_xcenter', 'bbox_ycenter', 'bbox_width', 'bbox_height'])
    df['xpred'] = round(df['bbox_xcenter']+(df['bbox_width']/2)).astype('int')
    df['ypred'] = round(df['bbox_ycenter']+df['bbox_width']).astype('int')
    df['pred'] = recog_result
    return df

def get_shape(img_ndarray):
    im_sz = img_ndarray.shape
    return im_sz[1], im_sz[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask api exposing number detection production code model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    app.run(host="0.0.0.0", port=5000)  # debug=True caused Restarting with stat    
