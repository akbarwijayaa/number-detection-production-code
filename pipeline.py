import os
import io
import time
import torch
import torch.backends.cudnn as cudnn
from os.path import dirname, join, abspath

import craftocr.test as test
import craftocr.imgproc as imgproc
import craftocr.file_utils as file_utils
import pandas as pd

from craftocr.craft import CRAFT
from collections import OrderedDict

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def detection(image_bytes, use_cuda = False, use_refine = True, use_poly = False,
              text_threshold = 0.9, link_threshold = 0.4, low_text = 0.4, 
              canvas_size = 1280, mag_ratio = 1.5, show_time = False):
    
    base_path = dirname(abspath(__file__))
    model_path = join(base_path, 'models')
    result_path = join(base_path, 'result')
    trained_model_path = join(model_path, 'craft_mlt_25k.pth')
    refine_path = join(model_path, 'craft_refiner_CTW1500.pth')
        
    data=pd.DataFrame(columns=['test','word_bboxes', 'pred_words', 'align_text'])
    data['test'] = ['test']
    # load net
    net = CRAFT()     # initialize

    if use_cuda:
        net.load_state_dict(test.copyStateDict(torch.load(trained_model_path)))
    else:
        net.load_state_dict(test.copyStateDict(torch.load(trained_model_path, map_location='cpu')))

    if use_cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if use_refine:
        from craftocr.refinenet import RefineNet
        refine_net = RefineNet()
        if use_cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(refine_path)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(refine_path, map_location='cpu')))

        refine_net.eval()
        use_poly = True

    t = time.time()
    # load data
    image = imgproc.loadImageFromNdArray(image_bytes)
    bboxes, polys, score_text, det_scores = test.test_net(net, image, text_threshold, link_threshold, low_text, use_cuda, use_poly, canvas_size, mag_ratio, show_time, refine_net)
    bbox_score={}

    for box_num in range(len(bboxes)):
        key = str (det_scores[box_num])
        item = bboxes[box_num]
        bbox_score[key]=item
    
    data['word_bboxes'][0]=bbox_score
    # print(bbox_score)   
    
    # save score texts
    # filename, file_ext = os.path.splitext(os.path.basename(image_path))
    # mask_file = result_path + "/res_" + filename + '_mask.jpg'
    # Save image result
    # cv2.imwrite(mask_file, score_text)
    # file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_path+'/')
    # print(data)
    return data, image
