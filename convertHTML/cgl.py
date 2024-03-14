from fsspec.core import url_to_fs
import json
from torch_geometric.data import Data
import torch
import os
from convertHTML.base import BaseDataset

class CGLDataset(BaseDataset):
    name = "cgl"
    labels =[
        "Logo",
        "Text",
        "Underlay",
        "Embellishment",
        "Highlighted text"
    ]
    def __init__(self,dir,split,max_seq_length,transform=None):
        super().__init__(dir,split,transform)
        self.N_category = self.num_classes
        self.dataset_name = "cgl"
    
    def process(self):
        data_list=[]
        raw_file = os.path.join(self.raw_dir,"train.json")
        fs,_=url_to_fs(self.raw_dir)
        with fs.open(raw_file,"rb") as f:
            json_t = json.load(f)
            ann = json_t['annotations']
            imgs = json_t['images']
            
            for ind,elements in enumerate(ann):
                img = imgs[ind]
                file_name = img['file_name']
                W = img['width']
                H = img['height']
                
                box = []
                label = []
                text = []
                for element in elements:
                    bbox = element['bbox']
                    x_m,y_m,w,h = bbox
                    xc = (x_m+w/2)
                    yc = (y_m+h/2)
                    
                    b = [round(xc/W,4),round(yc/H,4),round(w/W,4),round(h/H,4)]
                    
                    cat = element['category_id']
                    id = element['image_id']
                    te = element['text']
                    
                    box.append(b)
                    label.append(cat)
                    if te:
                        text.append(te)
                
                #boxes.append(torch.tensor(box,dtype=torch.float))
                #labels.append(torch.tensor(label,dtype=torch.long))
                #texts.append(text)
                text = " & ".join(text)
                attr = {
                    "name" : file_name,
                    "width" : W,
                    "height" : H,
                    "filtered" : False,
                    "has_canvas_element" : False,
                    "NoiseAdded" : False
                }
            
                data = Data(x=torch.tensor(box,dtype=torch.float),y=torch.tensor(label,dtype=torch.long),text = text)
                data.attr=attr
                data_list.append(data)
            generator = torch.Generator().manual_seed(0)
            indices = torch.randperm(len(data_list),generator=generator) # shuffling
            data_list = [data_list[i] for i in indices]
            N = len(data_list)
            print(N)
            s = [int(N*0.9),int(N*0.95)]

        with fs.open(self.processed_paths[0], "wb") as file_obj:
            torch.save(self.collate(data_list[: s[0]]), file_obj)
        with fs.open(self.processed_paths[1], "wb") as file_obj:
            torch.save(self.collate(data_list[s[0] : s[1]]), file_obj)
        with fs.open(self.processed_paths[2], "wb") as file_obj:
            torch.save(self.collate(data_list[s[1] :]), file_obj)