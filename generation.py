from src.model.minigpt4 import MiniGPT4
import os
import torch
from src.dataset.custom_dataset import RawFileDataset
from torch.utils.data import DataLoader
from PIL import Image
from src.processor.blip_processors import Blip2ImageTrainProcessor,Blip2ImageEvalProcessor
import random
from html_to_ui import *
 

import random
def build_input( canvas_width, canvas_height, vals) :
    '''
    vals : [(category, x, y, w, h), ...]
    '''
    
    def _build_rect(category='background', x=0, y=0, w=0, h=0):
        i = 0
        if x is None:
            x = '<FILL_{}>'.format(i)
            i += 1 
        if y is None:
            y = '<FILL_{}>'.format(i)
            i += 1 
        if w is None:
            w = '<FILL_{}>'.format(i)
            i += 1 
        if h is None:
            h = '<FILL_{}>'.format(i)
            i += 1 
        
        return f'<rect data-category=\"{category}\", x=\"{x}\", y=\"{y}\", width=\"{w}\", height=\"{h}\"/>\n'
    INSTRUCTION = [
        "I want to generate layout in poster design format.please generate the layout html according to the categories and image I provide"
        #"I want to generate layout in poster design format.please generate the layout html according to the categories and size and image I provide",
        #"I want to generate layout in poster design format.please recover the layout html according to the bbox, categories and size according to the image I provide"
    ]
    rects = ''
    for category, x, y, w, h in vals :
        rects += _build_rect(category, x, y, w, h)
    prompt = random.choice(INSTRUCTION)
    str_output = f'{prompt} (in html format):\n###bbox html:  <body> <svg width=\"{canvas_width}\" height=\"{canvas_height}\">{rects} </svg> </body>'
    return str_output


def inference_code(image,vals,model):

    width,height = image.size

    html_input = [build_input(width,height,vals)]

    img = image_processor(image).unsqueeze(0)
    
    with torch.no_grad():
        model.eval()
        with torch.autocast(device_type="cuda"):
            output = model.generate(img,html_input,max_new_tokens=400)
    return output[0], image


def generation_code(image,text,model):

    img = image_processor(image).unsqueeze(0)

    with torch.no_grad():
        model.eval()
        with torch.autocast(device_type="cuda"):
            output = model.generate(img,text,max_new_tokens=400)
    ex_bbox,ex_labels = get_bbox(output[0])
    img = draw_bbox(image, ex_bbox,ex_labels)
    return output, img




    



