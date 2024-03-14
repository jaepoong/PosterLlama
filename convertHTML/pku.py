from pandas import read_csv
import os
from fsspec.core import url_to_fs
import torch
import seaborn as sns
from pathlib import Path
from torch.utils.data import Dataset
import torch_geometric  
from torch_geometric.data import DataLoader  
from torch_geometric.data import Data
from convertHTML.base import BaseDataset

class PKUDataset(BaseDataset):
    name = "pku"
    labels =[
        "Text",
        "Logo",
        "Underlay",
    ]
    def __init__(self,dir,split,max_seq_length,transform=None):
        self.dir = dir
        self.img = os.listdir(os.path.join(self.dir,"train","inpainted_poster"))
        self.img = [item for item in self.img if '1739' not in item] # This index image do not have any layout, so make hesitation.
        self.poster_name = list(map(lambda x: "train/" + x.replace("_mask", ""), self.img))
        super().__init__(dir,split,transform)
        self.N_category = self.num_classes
        self.dataset_name = "pku"

    def process(self):
        data_list=[]
        W,H = 513,750
        class_list = [1.0, 2.0, 3.0]
        raw_file = os.path.join(self.dir,"train_csv_9973.csv")
        fs,_=url_to_fs(self.raw_dir)
        df = read_csv(raw_file)
        groups = df.groupby(df.poster_path)
        self.poster_name = [item for item in self.poster_name if 'train/processed' not in item]
        
        for i in range(len(self.poster_name)):
            sliced_df = groups.get_group(self.poster_name[i])
            cls = list(sliced_df["cls_elem"])
            box = list(map(eval, sliced_df["box_elem"]))
            box = [[round((bb[0]+bb[2])/2/W,4),round((bb[1]+bb[3])/2/H,4),round((bb[2]-bb[0])/W,4),round((bb[3]-bb[1])/H,4)] for bb in box]
            if 0 in cls:
                cls_mask = [element in class_list for element in cls]
                cls_mask_index = [index for index,element in enumerate(cls) if cls_mask[index]]
                cls = [cls[i] for i in cls_mask_index]
                box = [box[i] for i in cls_mask_index]
            name = self.poster_name[i].split("/")[-1]
            name = name.split('.')[0]+"_mask."+name.split('.')[1]
            
            attr = {
                "name" : name,
                "width" : 513,
                "height" : 750,
                "filtered" : False,
                "has_canvas_element" : False,
                "NoiseAdded" : False
                }
            
            data = Data(x= torch.tensor(box,dtype = torch.float), y= torch.tensor (cls, dtype=torch.int))
            data.attr = attr
            data_list.append(data)

        generator = torch.Generator().manual_seed(0)
        indices = torch.randperm(len(data_list),generator=generator) # shuffling
        data_list = [data_list[i] for i in indices]
        N = len(data_list)
        s = [int(N*0.8),int(N*0.9)]
        
        with fs.open(self.processed_paths[0], "wb") as file_obj:
            torch.save(self.collate(data_list[: s[0]]), file_obj)
        with fs.open(self.processed_paths[1], "wb") as file_obj:
            torch.save(self.collate(data_list[s[0] : s[1]]), file_obj)
        with fs.open(self.processed_paths[2], "wb") as file_obj:
            torch.save(self.collate(data_list[s[1] :]), file_obj)     