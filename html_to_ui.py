import sys
import os
os.environ["MKL_NUM_THREADS"] = "2" # export MKL_NUM_THREADS=2
os.environ["NUMEXPR_NUM_THREADS"] = "2" # export NUMEXPR_NUM_THREADS=2
os.environ["OMP_NUM_THREADS"] = "2" # export OMP_NUM_THREADS=2

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import torch
torch.set_num_threads(2)
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)  
import re
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import argparse
from collections import defaultdict
from helper.global_var import *
from typing import Dict
from tqdm import *
from PIL import Image, ImageDraw
import copy

DATASET_COLOR = {
    1: "#FF0000",   
    2: "#FFA500",   
    3: "#FFFF00",   
    4: "#008000",   
    5: "#0000FF"    
}
int_to_lable = DATASET_META.get("cgl")
label_to_int = dict([(v, k) for k, v in int_to_lable.items()])
def extract_WH(s):
    pattern = r'<svg width="([\d.]+)" height="([\d.]+)">'  
    match = re.search(pattern, s)  
    W, H = match.groups()  
    return  W, H  

def extract_xywh(s):
    bboxs, labels = [], []  
    pattern = r'<rect data-category="([^"]+)".*?x="(\d+(\.\d+)?)".*?y="(\d+(\.\d+)?)".*?width="(\d+(\.\d+)?)".*?height="(\d+(\.\d+)?)".*?/>'  
    matches = re.findall(pattern, s)  
    # print(matches)
    for match in matches:  
        data_category, x, _, y, _, width, _, height, _ = match  
        label = label_to_int.get(data_category)  
        # bbox = [x, y, x + width, y + height]  
        if label is None:
            continue
        bboxs.append([eval(x), eval(y), eval(width), eval(height)])
        labels.append(label)
    return bboxs, labels


def convert_to_array(generations):
    final_res = []
    for sample in generations:
        bboxs, cates = sample.get("bbox"), sample.get("categories")
        bboxs, cates = bboxs.tolist(), cates.tolist()
        bboxs = np.array(bboxs, dtype=np.float32)
        cates = np.array(cates, dtype=np.int32)
        final_res.append((bboxs, cates))
    return final_res
        

def average(scores):
    return sum(scores) / len(scores)


def draw_bbox(base_img, bbox, labels,verbose=True):
    img = copy.deepcopy(base_img)
    draw = ImageDraw.Draw(img)
    
    for b, l in zip(bbox, labels):
        if verbose:
            print("b, l : ", b, l)
        # draw.rectangle(b, outline=DATASET_COLOR[l], fill=DATASET_COLOR[l],  width=2)
        draw.rectangle([b[0],b[1] ,b[0]+b[2] ,b[1]+b[3]], outline=DATASET_COLOR[l], width=5)
    return img
    

def get_bbox(bbox):
    w, h = extract_WH(bbox)
    ex_bbox, ex_labels = extract_xywh(bbox)
    return ex_bbox, ex_labels
    
    
def main(args):
    data = pd.read_pickle(args.pkl_path)
    num_data = len(data['imgs'])
    
    for i in range(num_data):
        img = data['imgs'][i]
        bbox = data['labels'][i] #
        # print("bbox : ", bbox)
        ex_bbox, ex_labels = get_bbox(bbox)
        img = draw_bbox(img, ex_bbox, ex_labels)
        img.save("example/{:d}.png".format(i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate bounding box in image")
    # parser.add_argument("--img_path", type=str, action='store_true', default="", help="Advertisement Image")  
    # parser.add_argument("--html_path ", type=str, action='store_true', default="data/generated_results/rico", help='Generated HTML') 
    parser.add_argument("--dataset_name", type=str, default="cgl")
    parser.add_argument("--pkl_path", type=str, default="data/combined_data.pkl", help='pkl file') 
    args = parser.parse_args()
        
    int_to_lable = DATASET_META.get(args.dataset_name)
    label_to_int = dict([(v, k) for k, v in int_to_lable.items()])
    main(args)