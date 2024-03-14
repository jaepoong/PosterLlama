import sys
import os
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)  

import torch
import random
import torchvision.transforms as T
import os
import json
import copy
import argparse
from convertHTML.utils import LexicographicSort
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from transformers import AutoTokenizer
from convertHTML import get_dataset
from helper.global_var import *
from collections import OrderedDict
from typing import List, Dict  
from tqdm import *
import numpy as np
#from helper.metrics import *

##################
### Global Config
##################
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
SPAN_MASK_TOKEN = "<M>"
SEP_TOKEN = "<sep>"
PLACE_HOLDER = "<MASK>"


def round_nested_list(nested_list, decimals):  
    result = []  
    for item in nested_list:  
        if isinstance(item, list):   
            result.append(round_nested_list(item, decimals))  
        else:  
            result.append(round(item, decimals))  
    return result 

def add_gaussian_noise_and_resample(ele, x_max, y_max, sigma=0.01):
    def is_valid_bbox(bbox):
        return 0 <= bbox['x'] < x_max and 0 <= bbox['w'] < x_max  and 0 <= bbox['h'] < x_max and 0 <= bbox['y'] < y_max and 0 <= (bbox['x'] + bbox['w']) <= x_max and 1 <= (bbox['y'] + bbox['h']) <= y_max
    bbox = copy.deepcopy(ele)
    if not is_valid_bbox(ele):
        return ele
    def add_gaussian_noise(bbox):
        noise_x = np.random.normal(0, sigma)
        noise_y = np.random.normal(0, sigma)
        noise_w = np.random.normal(0, sigma)
        noise_h = np.random.normal(0, sigma)
        bbox['x'] = round(bbox['x']+(noise_x*x_max))
        bbox['y'] = round(bbox['y']+(noise_y*y_max))
        bbox['w'] = round(bbox['w']+(noise_w*x_max))
        bbox['h'] = round(bbox['h']+(noise_h*y_max))
        return bbox
    

    bbox = add_gaussian_noise(bbox)

    while not is_valid_bbox(bbox):
        bbox = copy.deepcopy(ele)
        add_gaussian_noise(bbox)

    return bbox

class CustomDataLoader(DataLoader):  
    def __init__(
        self, 
        args,
        tokenizer, 
        bbox_quantization, 
        dataset, 
        batch_size,
        shuffle=False, 
        split='train',
        **kwargs
    ):  
        super(CustomDataLoader, self).__init__(dataset, batch_size, shuffle, **kwargs)  
        self.split = split              # train, eval
        
        self.html_template = TEMPLATE_FORMAT.get("html_format")
        self.bbox_template = TEMPLATE_FORMAT.get("bbox_format")
        self.W=0
        self.H=0
        # self.text_template = TEMPLATE_FORMAT.get("text_format")
        
        if args.infilling:
            self.cond_cate_to_size_pos = INFILLING_INSTRUCTION.get("cond_cate_to_size_pos") # instruction + "bbox html:" html 
            self.cond_cate_size_to_pos = INFILLING_INSTRUCTION.get("cond_cate_size_to_pos")
            self.cond_random_mask = INFILLING_INSTRUCTION.get("cond_random_mask")
        else:
            self.cond_cate_to_size_pos = TEXT_INSTRUCTION.get("cond_cate_to_size_pos")
            self.cond_cate_size_to_pos = TEXT_INSTRUCTION.get("cond_cate_size_to_pos")
            self.cond_cate_pos_to_size = TEXT_INSTRUCTION.get("cond_cate_pos_to_size")
            self.cond_random_mask = TEXT_INSTRUCTION.get("cond_random_mask")
            self.unconditional = TEXT_INSTRUCTION.get("unconditional")
            self.refinement = TEXT_INSTRUCTION.get("refinement")
            self.completion = TEXT_INSTRUCTION.get("completion")
    
        if args.add_task_instruction:
            task_instruction = TASK_INSTRUCTION[args.dataset_name]
            self.cond_cate_to_size_pos = task_instruction + self.cond_cate_to_size_pos
            self.cond_cate_size_to_pos = task_instruction + self.cond_cate_size_to_pos
            self.cond_cate_pos_to_size = task_instruction + self.cond_cate_pos_to_size
            self.cond_random_mask = task_instruction + self.cond_random_mask
            self.unconditional = task_instruction + self.unconditional
            self.refinement = task_instruction + self.refinement
            self.completion = task_instruction + self.completion

        self.cond_bbox_prefix=""
        self.cond_cate_prefix=""
        self.category_map = DATASET_META[dataset.dataset_name]
        self.glue_template_train_eval = SEP_SEQ[0]
        self.glue_template_test = SEP_SEQ[1]
        self.glue_template_codegen_train = SEP_SEQ[2]
        self.glue_template_codegen_test = SEP_SEQ[3]
        
        self.tokenizer = tokenizer
        self.N_category = dataset.N_category
        self.bbox_quantization = bbox_quantization  # quanlization approachs
        self.consistency_num = args.consistency_num
        self.infilling = args.infilling
        
        
        
    def filter_invalid_num(self, lst, mask):
        new_lst = []
        for i in range(len(lst)):
            new_lst.append(lst[i][:mask[i].sum().item()])
        return new_lst
    
    
    def build_input_with_ele_dict(self, ele_dict: Dict, type=None):
        answer_notepad = []
        ele_dict = copy.deepcopy(ele_dict)
        if type == "html_content":
            ele_dict = ele_dict
        elif type == "cate_mask_html":
            answer_notepad = ele_dict["c"]
            ele_dict["c"] = PLACE_HOLDER
        elif type == "size_pos_mask_html":
            c = ele_dict["c"]
            answer_notepad = [ele_dict[k] for k in ele_dict if k != "c"]
            # ele_dict = dict([(k, PLACE_HOLDER) for k in ele_dict.keys()])
            ele_dict = {k: PLACE_HOLDER if k != 'content' else ele_dict[k] for k in ele_dict.keys()}
            if ele_dict.keys() == 'content':
                ele_dict
            ele_dict["c"] = c
        elif type == "size_mask_html":
            answer_notepad = [ele_dict["w"],ele_dict["h"]]
            ele_dict["w"] = PLACE_HOLDER
            ele_dict["h"] = PLACE_HOLDER
        elif type == "pos_mask_html":
            answer_notepad = [ele_dict["x"], ele_dict["y"]]
            ele_dict["x"] = PLACE_HOLDER
            ele_dict["y"] = PLACE_HOLDER
        elif type == "random_mask_html":
            random_mask_num = random.choice([3, 4]) # mask up to 80% places (categoty is not masked)
            selected_mask_element = random.sample(['x', 'y', 'w', 'h'], random_mask_num)
            answer_notepad = []
            for key in selected_mask_element:
                answer_notepad.append(ele_dict[key])
                ele_dict[key] = PLACE_HOLDER
        elif type == "refinement_html":
            ele_dict = add_gaussian_noise_and_resample(ele_dict,self.W,self.H)
            
        return self.bbox_template.format(**ele_dict), answer_notepad
    
    
    def replace_order_mask(self, lst: List[str], ans_lst: List):
        '''
        replace the mask token and build corresponding results
        '''
        new_lst, new_ans = [], {}
        cnt = 1
        for line, ans in zip(lst, ans_lst):
            mask_count = line.count(PLACE_HOLDER)
            for i in range(mask_count):
                mask_token = SPAN_MASK_TOKEN.format(i=cnt)
                line = line.replace(PLACE_HOLDER, mask_token, 1)
                new_ans[mask_token] = ans[i]
                cnt += 1
            new_lst.append(line)
        return new_lst, new_ans
    
    def convert_num_to_html(self, coord_lst=None, category_lst=None, self_consistency=False, consistency_num=10):
    # def convert_num_to_html(self, coord_lst=None, category_lst=None, self_consistency=False, consistency_num=10):
        batched_html_lst = []  # target
        batched_cond_cate, batched_cond_bbox = [], []  # condition
        unconditional_ans=[]
        unconditional, refinement, random_mask, completion = [""], [], [], []
        cond_cate_to_size_pos, cond_cate_size_to_pos, cond_cate_pos_to_size = [], [], [] 

        # text_len = [i for i in range(len(text))]
        if coord_lst is not None and category_lst is not None: # create the training data   
            for coords, categories in zip(coord_lst, category_lst):
                #print(coords)
                # store all the input code
                html_content = []
                cate_mask_html, random_mask_html = [], []
                unconditional_html, refinement_html, completion_html=[],[],[]
                size_pos_mask_html, pos_mask_html, size_mask_html = [], [], []
                
                # store all the ans
                cate_mask_html_ans, random_mask_html_ans = [], []
                unconditional_html_ans, refinement_html_ans, completion_html_ans = [], [], []
                size_pos_mask_html_ans, pos_mask_html_ans, size_mask_html_ans = [], [], []
                
                all_category = OrderedDict([(i, 0) for i in range(self.N_category)])
                i = 0
                for coord, category in zip(coords, categories):
                    #content = text[0][i]
                    w, h = int(coord[2]), int(coord[3]) 
                    x, y = int(coord[0] - w / 2), int(coord[1] - h / 2) # c->xl, c->yl
                    real_category = self.category_map[category]
                    all_category[category] += 1
                    ele_dict = {"c": real_category, "x": x, "y":y, "w":w, "h":h}
                    #ele_dict = {"c": real_category, "x": x, "y":y, "w":w, "h":h, "content":content}
                    tmp1, _ = self.build_input_with_ele_dict(ele_dict, "html_content")
                    html_content.append(tmp1)
                    
                    # category mask to PLACE_HOLDER
                    tmp2, ans2 = self.build_input_with_ele_dict(ele_dict, "cate_mask_html") 
                    cate_mask_html.append(tmp2)
                    cate_mask_html_ans.append(ans2)
                    # random_mask_html
                    tmp3, ans3 = self.build_input_with_ele_dict(ele_dict, "random_mask_html")
                    random_mask_html.append(tmp3)
                    random_mask_html_ans.append(ans3)
                    # unconditional_html
                    #tmp4, ans4 = self.build_input_with_ele_dict(ele_dict, "unconditional_html")
                    #unconditional_html.append(tmp4)
                    #unconditinoal_html_ans.append(ans4)
                    # refinement_html
                    tmp5, ans5 = self.build_input_with_ele_dict(ele_dict, "refinement_html")
                    refinement_html.append(tmp5)
                    #refinement_html_ans.append(ans5)

                    # size_pos_mask_html
                    tmp6, ans6 = self.build_input_with_ele_dict(ele_dict, "size_pos_mask_html")
                    size_pos_mask_html.append(tmp6)
                    size_pos_mask_html_ans.append(ans6)
                    # pos_mask_html
                    tmp7, ans7 = self.build_input_with_ele_dict(ele_dict, "pos_mask_html")
                    pos_mask_html.append(tmp7)
                    pos_mask_html_ans.append(ans7)
                    # size_mask_html
                    tmp8, ans8 = self.build_input_with_ele_dict(ele_dict, "size_mask_html")
                    size_mask_html.append(tmp8)
                    size_mask_html_ans.append(ans8)
                    # completion_html
                    #tmp9, ans9 = self.build_input_with_ele_dict(ele_dict, "completion")
                    #size_mask_html.append(tmp9)
                    #size_mask_html_ans.append(ans9)        

                    
                    i += 1
                ### post process the mask token id
                cate_mask_html, cate_mask_ans = self.replace_order_mask(cate_mask_html, cate_mask_html_ans)
                random_mask_html, random_mask_ans = self.replace_order_mask(random_mask_html, random_mask_html_ans)
                size_pos_mask_html, size_pos_mask_ans = self.replace_order_mask(size_pos_mask_html, size_pos_mask_html_ans)
                pos_mask_html, pos_mask_ans = self.replace_order_mask(pos_mask_html, pos_mask_html_ans)
                size_mask_html,size_mask_ans = self.replace_order_mask(size_mask_html,size_mask_html_ans)
                
                verbal_all_categories = []
                for i in range(self.N_category):
                    if all_category[i] != 0:
                        verbal_category = self.category_map[i]
                        verbal_number = VERBALIZED_NUM[all_category[i]]
                        verbal_all_categories.append("{} {},".format(verbal_number, verbal_category))
                all_verbal_all_cates = " ".join(verbal_all_categories).rstrip(",")
                
                if self_consistency == True:  # random shuffle the condition, but stay with the target
                    shuffle_lst = [i for i in range(len(html_content))]
                    min_shuffle_num = min(len(shuffle_lst), consistency_num) # min(gt contents개수, consistency num)
                    
                    def shuffle_list(input_list):  
                        random.shuffle(input_list)  
                        return input_list  
                    
                    shuffled_results = []  
                    for i in range(min_shuffle_num): # 순서 섞기.
                        shuffled_results.append(shuffle_list(shuffle_lst.copy()))
                    
                    for random_order in shuffled_results:
                        new_html_content = [html_content[i] for i in random_order]
                        unconditional_html = [html_content[i] for i in random_order] 
                        new_cate_mask_html = [cate_mask_html[i] for i in random_order]
                        new_size_pos_mask_html = [size_pos_mask_html[i] for i in random_order]
                        new_pos_mask_html = [pos_mask_html[i] for i in random_order]
                        new_size_mask_html = [size_mask_html[i] for i in random_order]
                        new_random_mask_html = [random_mask_html[i] for i in random_order]
                        new_completion_html = [html_content[i] for i in random_order]
                        new_refinement = [refinement_html[i] for i in random_order]

                        batched_cond_cate.append(all_verbal_all_cates)
                        batched_html_lst.append("\n".join(new_html_content))  # save target
                        
                        unconditional_ans.append("\n".join(unconditional_html))
                        batched_cond_bbox.append('\n'.join(new_cate_mask_html))
                        cond_cate_to_size_pos.append("\n".join(new_size_pos_mask_html)) #c ->sp
                        cond_cate_size_to_pos.append("\n".join(new_pos_mask_html)) # cs -> p
                        cond_cate_pos_to_size.append("\n".join(new_size_mask_html)) # cp -> s
                        random_mask.append("\n".join(new_random_mask_html)) 
                        completion_html_ans.append("\n".join(new_completion_html))
                        extract_index = random.randint(1,len(random_order))
                        completion_html.append("\n".join(new_completion_html[:extract_index]))
                        refinement.append("\n".join(new_refinement))
                        
                else:
                    # process all conditions
                    batched_cond_cate.append(all_verbal_all_cates)  
                    unconditional_ans.append('\n'.join(html_content)) 
                    batched_cond_bbox.append('\n'.join(cate_mask_html))
                    batched_html_lst.append("\n".join(html_content))
                    cond_cate_to_size_pos.append("\n".join(size_pos_mask_html))
                    cond_cate_size_to_pos.append("\n".join(pos_mask_html))
                    cond_cate_pos_to_size.append("\n".join(size_mask_html))
                    random_mask.append("\n".join(random_mask_html))
                
        else:
            raise ValueError("Can not inplement to testing data")
        return {
            "batched_html_lst": batched_html_lst,
            "batched_cond_cate": batched_cond_cate,
            "batched_cond_bbox": batched_cond_bbox,
            "cond_cate_to_size_pos": cond_cate_to_size_pos,
            "cond_cate_size_to_pos": cond_cate_size_to_pos,
            "cond_cate_pos_to_size" : cond_cate_pos_to_size,
            "random_mask": random_mask,
            "unconditional" : unconditional*len(random_mask),
            "completion" : completion_html,
            "refinement" : refinement,
            "codegen_ans": {
                "cate_mask_ans": cate_mask_ans,
                "size_pos_mask_ans": size_pos_mask_ans,
                "unconditional_ans": unconditional_ans,
                "pos_mask_ans": pos_mask_ans,
                "size_mask_ans": size_mask_ans,
                "random_mask_ans": random_mask_ans,
                "completion_ans" : completion_html_ans
            },
        }
    
    
    def build_random_mask(self, lst):
        new_lst = lst.copy()
        num = random.sample([3, 4], 1)[0]  # mask up to 80% position
        pos = random.sample([0,1,2,3], num)
        for i in pos:
            new_lst[i] = PLACE_HOLDER
        return new_lst
    
    
    def generate_new_order(self, lst):
        shuffle_order = [i for i in range(len(lst))]
        random.shuffle(shuffle_order)
        return shuffle_order
        
    def custom_function(self, data, id_, self_consistency=True, consistency_num=10): 
        label, mask = to_dense_batch(data.y, data.batch)   # (B, S)
  
        bbox_real, _ = to_dense_batch(data.x, data.batch)  # (B, S, 4)
        ####################### 
        text = data.text
        text = text[0]
        ####################### 
        W, H,name = data.attr["width"], data.attr["height"],data.attr["name"] #name
        self.W=W[0].item()
        self.H=H[0].item()
        size_ = torch.cat((W.unsqueeze(-1), H.unsqueeze(-1), W.unsqueeze(-1), H.unsqueeze(-1)), dim=-1)
        size_ = size_.unsqueeze(1)
        real_idx = size_ * bbox_real # [cx, cy, w, h]
        if self.bbox_quantization == "code":
            label = label.to(torch.int).tolist()
            label_lst = self.filter_invalid_num(label, mask)        # [[2, 2, 3, 2]]
            real_idx = real_idx.to(torch.float).tolist()
            
            real_idx = round_nested_list(real_idx, 1)
            bbox_lst = self.filter_invalid_num(real_idx, mask)      # 0:[[258.0, 72.5, 400.0, 61.0], [257.5, 134.5, 299.0, 33.0], [257.5, 696.5, 169.0, 37.0], [256.5, 695.5, 113.0, 25.0]]
            preposed_res = self.convert_num_to_html(bbox_lst, label_lst, self_consistency=self_consistency, consistency_num=consistency_num)
            batched_html_lst = preposed_res.get("batched_html_lst")
            
            batched_cond_cate = preposed_res.get("batched_cond_cate")
            batched_cond_bbox = preposed_res.get("batched_cond_bbox")   
            
            cond_cate_to_size_pos = preposed_res.get("cond_cate_to_size_pos")
            cond_cate_to_size_pos_res_dict = preposed_res["codegen_ans"].get("size_pos_mask_ans")
            
            cond_cate_size_to_pos = preposed_res.get("cond_cate_size_to_pos")
            cond_cate_size_to_pos_res_dict = preposed_res["codegen_ans"].get("pos_mask_ans")
            
            cond_cate_pos_to_size = preposed_res.get("cond_cate_pos_to_size")
            cond_cate_pos_to_size_res_dict = preposed_res["codegen_ans"].get("size_mask_ans")

            unconditional = preposed_res.get("unconditional")
            unconditional_ans = preposed_res['codegen_ans'].get("unconditional_ans")
            
            random_mask = preposed_res.get("random_mask")
            random_mask_res_dict = preposed_res["codegen_ans"].get("random_mask_ans")
            
            completion = preposed_res.get("completion")
            completion_ans = preposed_res["codegen_ans"].get("completion_ans")

            refinement = preposed_res.get("refinement")
        
        else:
            raise Exception("We only developed on code format yet.")


        if self_consistency:  # resize W and H
            W = W.repeat(len(batched_html_lst))
            H = H.repeat(len(batched_html_lst))
        
        # construct the html input 
        batched_cond_bbox = [
            self.html_template.format(W=W[i], H=H[i], content=batched_cond_bbox[i])
            for i in range(len(batched_cond_bbox))                 
        ]
        cond_cate_to_size_pos = [
            self.html_template.format(W=W[i], H=H[i], content=cond_cate_to_size_pos[i])
            for i in range(len(cond_cate_to_size_pos))  
        ]
        cond_cate_size_to_pos = [
            self.html_template.format(W=W[i], H=H[i], content=cond_cate_size_to_pos[i])
            for i in range(len(cond_cate_size_to_pos))  
        ]
        cond_cate_pos_to_size = [
            self.html_template.format(W=W[i], H=H[i], content=cond_cate_pos_to_size[i])
            for i in range(len(cond_cate_pos_to_size))
        ]
        unconditional = [
            self.html_template.format(W=W[i], H=H[i], content=unconditional[i])
            for i in range(len(unconditional))
        ]
        cond_recover_mask = [
            self.html_template.format(W=W[i], H=H[i], content=random_mask[i])
            for i in range(len(random_mask))  
        ]
        completion = [
            self.html_template.format(W=W[i], H=H[i], content=completion[i])
            for i in range(len(completion))
        ]
        refinement = [
            self.html_template.format(W=W[i], H=H[i], content=refinement[i])
            for i in range(len(refinement))
        ]
        
        # add task instructions make html format.
        cond_recover_mask = [
            self.cond_random_mask.format(text=text,bbox_html=bbox)
            for bbox in cond_recover_mask
        ]
        unconditional = [
            self.unconditional.format(text=text,bbox_html=bbox)
            for bbox in unconditional
        ]
        cond_cate_to_size_pos = [
            self.cond_cate_to_size_pos.format(text=text,bbox_html=bbox)
            for bbox in cond_cate_to_size_pos
        ]
        cond_cate_size_to_pos = [
            self.cond_cate_size_to_pos.format(text=text,bbox_html=bbox)
            for bbox in cond_cate_size_to_pos
        ]
        cond_cate_pos_to_size = [
            self.cond_cate_pos_to_size.format(text=text,bbox_html=bbox)
            for bbox in cond_cate_pos_to_size
        ]
        completion = [
            self.completion.format(text=text,bbox_html=bbox)
            for bbox in completion
        ]
        refinement = [
            self.refinement.format(text=text,bbox_html=bbox)
            for bbox in refinement
        ]
        
        bbox_cond_seqs = [
            self.cond_bbox_prefix.format(categories=cate, bbox_html=bbox_html) 
            for cate, bbox_html in zip(batched_cond_cate, batched_cond_bbox)
        ]

        category_cond_seqs = [
            self.cond_cate_prefix.format(categories=batched_cond_cate[i], W=W[i], H=H[i]) 
            for i in range(len(batched_cond_cate))
        ]

        if self.infilling and self.split in ("train", "val"):  # do infilling task
            cond_cate_to_size_pos_golden = [f" {SEP_TOKEN} ".join(f"{key} {value}" for key, value in cond_cate_to_size_pos_res_dict.items())]
            cond_cate_size_to_pos_golden = [f" {SEP_TOKEN} ".join(f"{key} {value}" for key, value in cond_cate_size_to_pos_res_dict.items())]
            cond_cate_pos_to_size_golden = [f" {SEP_TOKEN} ".join(f"{key} {value}" for key, value in cond_cate_pos_to_size_res_dict.items())]
            random_mask_res_dict_golden = [f" {SEP_TOKEN} ".join(f"{key} {value}" for key, value in random_mask_res_dict.items())]
        
        # build target seq
        if self.split == "train" or self.split == "val":
            if self.infilling:
                if self_consistency:
                    consistency_num = len(cond_cate_to_size_pos)
                    target_seqs = [
                        cond_cate_to_size_pos_golden * consistency_num, 
                        cond_cate_size_to_pos_golden * consistency_num, 
                        cond_cate_pos_to_size_golden * consistency_num,
                        random_mask_res_dict_golden * consistency_num
                    ]
                else:
                    target_seqs = [cond_cate_to_size_pos_golden, cond_cate_size_to_pos_golden, cond_cate_pos_to_size_golden, random_mask_res_dict_golden]
                
                cond_cate_to_size_pos_seq_modeling = [
                    self.glue_template_codegen_train.format(instruct=sample, result=target)
                    for sample, target in zip(cond_cate_to_size_pos, target_seqs[0])
                ]
                
                cond_cate_size_to_pos_seq_modeling = [
                    self.glue_template_codegen_train.format(instruct=sample, result=target)
                    for sample, target in zip(cond_cate_size_to_pos, target_seqs[1])
                ]
                
                cond_cate_pos_to_size_modeling = [
                    self.glue_template_codegen_train.format(instruct=sample, result=target)
                    for sample, target in zip(cond_cate_pos_to_size, target_seqs[2])
                ]

                
                cond_recover_mask_seq_modeling = [
                    self.glue_template_codegen_train.format(instruct=sample, result=target)
                    for sample, target in zip(cond_recover_mask, target_seqs[3])
                ]

                return {
                    "cond_cate_to_size_pos_seq_modeling": cond_cate_to_size_pos_seq_modeling,
                    "cond_cate_size_to_pos_seq_modeling": cond_cate_size_to_pos_seq_modeling,
                    "cond_cate_pos_to_size_seq_modeling" : cond_cate_pos_to_size_seq_modeling,
                    "cond_recover_mask_seq_modeling": cond_recover_mask_seq_modeling,
                    "name" : name #
                }
                
            else:
                target_seqs = [
                    self.html_template.format(W=W[i], H=H[i], content=batched_html_lst[i])
                    for i in range(W.size(0))
                ]
            
                cond_recover_mask_seq_modeling = [
                    self.glue_template_codegen_train.format(instruct=sample, result=target)
                    for sample, target in zip(cond_recover_mask, target_seqs)
                ]
                unconditional_seq_modeling = [
                    self.glue_template_codegen_train.format(instruct=sample, result=target)
                    for sample, target in zip(unconditional, target_seqs)
                ]
                
                cond_cate_to_size_pos_seq_modeling = [
                    self.glue_template_codegen_train.format(instruct=sample, result=target)
                    for sample, target in zip(cond_cate_to_size_pos, target_seqs)
                ]
                
                cond_cate_size_to_pos_seq_modeling = [
                    self.glue_template_codegen_train.format(instruct=sample, result=target)
                    for sample, target in zip(cond_cate_size_to_pos, target_seqs)
                ]
                cond_cate_pos_to_size_seq_modeling = [
                    self.glue_template_codegen_train.format(instruct=sample, result=target)
                    for sample, target in zip(cond_cate_pos_to_size, target_seqs)
                ]
                refinement_seq_modeling= [
                    self.glue_template_codegen_train.format(instruct=sample, result=target)
                    for sample, target in zip(refinement, target_seqs)
                ]
                target_seqs = [
                    self.html_template.format(W=W[i], H=H[i], content=completion_ans[i])
                    for i in range(W.size(0))
                ]
                completion_seq_modeling =[
                    self.glue_template_codegen_train.format(instruct=sample, result=target)
                    for sample, target in zip(completion, target_seqs)
                ]

                # print(cond_cate_size_to_pos_seq_modeling)
                return {
                    "cond_cate_to_size_pos_seq_modeling": cond_cate_to_size_pos_seq_modeling,
                    "cond_cate_size_to_pos_seq_modeling": cond_cate_size_to_pos_seq_modeling,
                    "cond_cate_pos_to_size_seq_modeling" : cond_cate_pos_to_size_seq_modeling,
                    "unconditional_seq_modeling" : unconditional_seq_modeling,
                    "cond_recover_mask_seq_modeling": cond_recover_mask_seq_modeling,
                    "completion_seq_modeling" : completion_seq_modeling,
                    "refinement_seq_modeling" : refinement_seq_modeling,
                    "name" : name #s
                }
            
        else:
            if self.infilling:
                cond_bbox_input_seqs = [
                    self.glue_template_codegen_test.format(instruct=instance)
                    for instance in bbox_cond_seqs
                ]
                
                continual_gen_input_seqs = [
                    self.glue_template_codegen_test.format(instruct=instance)
                    for instance in category_cond_seqs
                ]
                
                cond_cate_size_to_pos_input_seqs = [
                    self.glue_template_codegen_test.format(instruct=instance)
                    for instance in cond_cate_size_to_pos
                ]

                cond_cate_pos_to_size_input_seqs = [
                    self.glue_template_codegen_test.format(instruct=instance)
                    for instance in cond_cate_pos_to_size
                ]
                
                cond_cate_to_size_pos_input_seqs = [
                    self.glue_template_codegen_test.format(instruct=instance)
                    for instance in cond_cate_to_size_pos
                ]
                
                cond_recover_mask_input_seqs = [
                    self.glue_template_codegen_test.format(instruct=instance)
                    for instance in cond_recover_mask
                ]
            
            else:
                cond_bbox_input_seqs = [
                    self.glue_template_codegen_test.format(instruct=instance)
                    for instance in bbox_cond_seqs
                ]
                
                continual_gen_input_seqs = [
                    self.glue_template_codegen_test.format(instruct=instance)
                    for instance in category_cond_seqs
                ]
                
                cond_cate_size_to_pos_input_seqs = [
                    self.glue_template_codegen_test.format(instruct=instance)
                    for instance in cond_cate_size_to_pos
                ]

                cond_cate_pos_to_size_input_seqs = [
                    self.glue_template_codegen_test.format(instruct=instance)
                    for instance in cond_cate_pos_to_size
                ]
                
                cond_cate_to_size_pos_input_seqs = [
                    self.glue_template_codegen_test.format(instruct=instance)
                    for instance in cond_cate_to_size_pos
                ]
                unconditional_input_seqs = [
                    self.glue_template_codegen_test.format(instruct=instance)
                    for instance in unconditional
                ]
                
                cond_recover_mask_input_seqs = [
                    self.glue_template_codegen_test.format(instruct=instance)
                    for instance in cond_recover_mask
                ]
                completion_input_seqs=[
                    self.glue_template_codegen_test.format(instruct=instance)
                    for instance in completion 
                ]
                refinement_input_seqs = [
                    self.glue_template_codegen_test.format(instruct=instance)
                    for instance in refinement 
                ]

            
            
            labels = None
            if batched_html_lst is not None:
                
                labels = [
                    self.html_template.format(W=W[i], H=H[i], content=batched_html_lst[i])
                    for i in range(W.size(0))
                ]

            return {
                "cond_bbox_input_seqs": cond_bbox_input_seqs,
                "continual_gen_input_seqs": continual_gen_input_seqs,
                "cond_cate_size_to_pos_input_seqs": cond_cate_size_to_pos_input_seqs,
                "cond_cate_pos_to_size_input_seqs" : cond_cate_pos_to_size_input_seqs,
                "cond_cate_to_size_pos_input_seqs": cond_cate_to_size_pos_input_seqs,
                "unconditional_input_seqs" : unconditional_input_seqs,
                "cond_recover_mask_input_seqs": cond_recover_mask_input_seqs,
                "completion_input_seqs" : completion_input_seqs,
                "refinement_input_seqs" : refinement_input_seqs,
                "labels": labels,
                "name" : name,
                "raw_data": {
                    "category":label_lst[0] ,
                    "bbox": bbox_lst[0],
                },
                "id_": id_
            }
    
    def __iter__(self): 
        for i, data in enumerate(super(CustomDataLoader, self).__iter__()):  
            if self.consistency_num > 1:
                self_consistency = True
            else:
                self_consistency = False
            yield self.custom_function(data, i, self_consistency=self_consistency)  

    
    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id
    
    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_token_id
    
    @property
    def pad_token_id(self) -> int: 
        return self.tokenizer.pad_token_id
    
    @property
    def mask_token_id(self) -> int:
        return self.tokenizer.unk_token_id
    
    @property
    def unk_token_id(self) -> int:
        return self.tokenizer.unk_token_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the Layout Code in HTML format")
    parser.add_argument("--dataset_name", type=str, default="rico25", help="dataset name")  
    parser.add_argument("--dataset_path", type=str, default="data/rico25-max25")
    parser.add_argument("--save_path", type=str, default="data/rico25-max25/html_format")
    parser.add_argument("--model_path_or_name", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="tokenizer model name")
    parser.add_argument("--bbox_quantization", type=str, default="code", choices=["code", "numerical"])
    parser.add_argument("--consistency_num", type=int, default=1, help="number of consistency num")
    parser.add_argument("--build_testing_set", action="store_true", help="whether to build the testing set")
    parser.add_argument("--infilling", action="store_true", help="whether to build the infilling data set")
    parser.add_argument("--add_task_instruction", action="store_true", help="whether to add the task instruction")
    args = parser.parse_args()
    
    if not os.path.exists(args.save_path):  
        os.makedirs(args.save_path)  
        print(f"Directory '{args.save_path}' created.")  
    
    if args.model_path_or_name is None:
        print("please specify the model name or path")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_name)
    transforms = [LexicographicSort()]
     
    if not args.build_testing_set:
        train_dataset = get_dataset(
            name=args.dataset_name,
            datapath=args.dataset_path,
            split='train',
            #transform=T.Compose(transforms)
        )
        
        eval_dataset = get_dataset(
            name=args.dataset_name,
            datapath=args.dataset_path,
            split='val',
            #transform=T.Compose(transforms)
        )
        
        train_dataloader = CustomDataLoader(
            args,
            tokenizer,
            bbox_quantization=args.bbox_quantization,
            dataset=train_dataset, 
            batch_size=1,
            split="train",
        )
        
        eval_dataloader = CustomDataLoader(
            args,
            tokenizer,
            bbox_quantization=args.bbox_quantization,
            dataset=eval_dataset, 
            batch_size=1,
            split="val",
        )
        print(len(train_dataset))
        all_train_data, all_eval_data = [], []
    
        train_file = os.path.join(args.save_path, "train_llama_numerical.jsonl")
        val_file = os.path.join(args.save_path, "val_llama_numerical.jsonl")
        
        print(f"begin to save train file >>> {args.save_path}")
        with tqdm(total=len(train_dataloader)) as pbar:
            for i, batch_inputs in enumerate(train_dataloader):
                if args.consistency_num > 1:
                    inner_batch = len(batch_inputs['cond_cate_to_size_pos_seq_modeling'])
                    batch_inputs['name'] = batch_inputs['name']*inner_batch
                    new_batch_inputs = [{} for i in range(inner_batch)]
                    for k, v in batch_inputs.items():
                        for i, value in enumerate(v):
                            new_batch_inputs[i][k] = value    
                    batch_inputs = new_batch_inputs
                else:
                    batch_inputs = [batch_inputs]
                all_train_data.extend(batch_inputs)
                pbar.update(1)
        with open(train_file, "w") as f:
            for line in all_train_data:
                f.write(json.dumps(line) + "\n")
                
        print(f"training data saved done, begin to save eval dataset >>> {args.save_path}")
        with tqdm(total=len(eval_dataloader)) as pbar:
            for i, batch_inputs in enumerate(eval_dataloader):
                if args.consistency_num > 1:
                    inner_batch = len(batch_inputs['cond_cate_to_size_pos_seq_modeling'])
                    batch_inputs['name'] = batch_inputs['name']*inner_batch
                    new_batch_inputs = [{} for i in range(inner_batch)]
                    for k, v in batch_inputs.items():
                        for i, value in enumerate(v):
                            new_batch_inputs[i][k] = value    
                    batch_inputs = new_batch_inputs
                else:
                    batch_inputs = [batch_inputs]
                all_eval_data.extend(batch_inputs)
                pbar.update(1)
        with open(val_file, "w") as f:
            for line in all_eval_data:
                f.write(json.dumps(line) + "\n")
    
    else:
        test_dataset = get_dataset(
            name=args.dataset_name,
            datapath=args.dataset_path,
            split='test',
            transform=T.Compose(transforms)
        )
        
        test_dataloader = CustomDataLoader(
            args,
            tokenizer,
            bbox_quantization=args.bbox_quantization,
            dataset=test_dataset, 
            batch_size=1,
            split="test",
        )
    
        all_test_data = []
        test_file = os.path.join(args.save_path, "test_numerical.jsonl")
        print("begin to save test file")
        with tqdm(total=len(test_dataloader)) as pbar:
            for i, batch_inputs in enumerate(test_dataloader):
                all_test_data.append(batch_inputs)
                pbar.update(1)
        with open(test_file, "w") as f:
            for line in all_test_data:
                f.write(json.dumps(line) + "\n")
    
    
    