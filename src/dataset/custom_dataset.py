from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, AutoImageProcessor
import json
import random
import torch
import transformers
from PIL import Image
import os
from src.processor import Blip2ImageTrainProcessor,Blip2ImageEvalProcessor


def split_string_by_delimiter(input_string, delimiter):

    parts = input_string.split(delimiter)

    if len(parts) >= 2:
        before_delimiter = parts[0]
        after_delimiter = delimiter.join(parts[1:])
        return before_delimiter, after_delimiter
    else:
        return input_string, ""

def process_input(a,aug=False):
    if isinstance(a, list):
        if len(a) > 0:
            if aug:
                selected_element = random.choice(a)
                return selected_element
            else:
                selected_element = a[0]
                return selected_element
        else:
            return None  
    elif isinstance(a, str):
        return a
    else:
        raise ValueError("Unsupported input type")

class RawFileDataset(Dataset):  
    def __init__(self, file,img_file_path ="data/cgl_dataset/augment_cgl",image_processor = Blip2ImageEvalProcessor(),vit_model_name = None,aug=False):  

        with open(file, "r") as f:
            self.content = [json.loads(line) for line in f]
        self.cond_type = ["cond_cate", "cond_cate_size", "cond_cate_pos", "unconditional", "recover", "completion","refinement"]
        self.sample_prob = [1, 1, 1, 2, 2, 2, 2]  # custom your sampling weight here
        self.img_file_path = img_file_path #
        self.image_processor = image_processor
        if vit_model_name == "dino_v2":
            self.image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        
        self.vit_model_name = vit_model_name
        self.aug = aug
    def __len__(self):
        return len(self.content)

    def __getitem__(self, index):
        data = self.content[index]

        cond_cate_to_size_pos_seq_modeling = data["cond_cate_to_size_pos_seq_modeling"]
        cond_cate_size_to_pos_seq_modeling = data["cond_cate_size_to_pos_seq_modeling"]
        cond_cate_pos_to_size_seq_modeling = data["cond_cate_pos_to_size_seq_modeling"]
        unconditional_seq_modeling = data["unconditional_seq_modeling"]
        cond_recover_mask_seq_modeling = data["cond_recover_mask_seq_modeling"]
        completion_seq_modeling = data["completion_seq_modeling"]#
        refinement_seq_modeling = data["refinement_seq_modeling"]#
        name = process_input(data["name"],aug=self.aug) # if list output random element, str output corresponding str
        
        if self.img_file_path :
            try: 
                img = Image.open(os.path.join(self.img_file_path,name[:-4]+".png"))
            except: 
                img = Image.open(os.path.join(self.img_file_path,name[:-4]+".jpg"))
        else:
            img = None
        
        if self.image_processor:
            if self.vit_model_name =="dino_v2":
                img = torch.tensor(self.image_processor(img)['pixel_values'][0])
            else:
                img = self.image_processor(img)
            
        
        instruct_1,answer_1 = split_string_by_delimiter(cond_cate_to_size_pos_seq_modeling,"<MID>")
        instruct_2,answer_2 = split_string_by_delimiter(cond_cate_size_to_pos_seq_modeling,"<MID>")
        instruct_3,answer_3 = split_string_by_delimiter(cond_cate_pos_to_size_seq_modeling,"<MID>")
        instruct_4,answer_4 = split_string_by_delimiter(unconditional_seq_modeling,"<MID>")
        instruct_5,answer_5 = split_string_by_delimiter(cond_recover_mask_seq_modeling,"<MID>")
        instruct_6,answer_6 = split_string_by_delimiter(completion_seq_modeling,"<MID>")
        instruct_7,answer_7 = split_string_by_delimiter(refinement_seq_modeling,"<MID>")

        instances = {
            "cond_cate":{
                "input":  instruct_1.strip(),
                "labels": answer_1.strip(),
                "image" : img,
                "name" : name
            },
            "cond_cate_size": {
                "input": instruct_2.strip(),
                "labels": answer_2.strip(),
                "image" : img,
                "name" : name
            },
            "cond_cate_pos": {
                "input": instruct_3.strip(),
                "labels": answer_3.strip(),
                "image" : img,
                "name" : name
            },
            "unconditional": {
                "input": instruct_4.strip(),
                "labels": answer_4.strip(),
                "image" : img,
                "name" : name
            },
            "recover": {
                "input": instruct_5.strip(),
                "labels": answer_5.strip(),
                "image" : img,
                "name" : name
            },
            "completion": {
                "input": instruct_6.strip(),
                "labels": answer_6.strip(),
                "image" : img,
                "name" : name
            },      
            "refinement": {
                "input": instruct_7.strip(),
                "labels": answer_7.strip(),
                "image" : img,
                "name" : name
            }
            }

        selected_types = random.choices(self.cond_type, self.sample_prob, k=1)  # joint loss (t=2 here)
        instance = instances[selected_types[0]]
        return instance