import torch
import torch.nn as nn
import numpy as np
import multiprocessing as mp
from itertools import chain
from scipy.optimize import linear_sum_assignment
#from util import convert_xywh_to_ltrb
from pytorch_fid.fid_score import calculate_frechet_distance
from collections import defaultdict

class LayoutFID():
    def __init__(self, pth, device='cpu'):
        num_label = 13 #if 'rico' in pth or 'enrico' in pth or 'clay' in pth or 'ads_banner_collection' in pth or 'AMT_uploaded_ads_banners' in pth or 'cgl_dataset' in pth else 5
        self.model = LayoutNet(num_label).to(device)

        # load pre-trained LayoutNet
        state_dict = torch.load(pth, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.requires_grad_(False)
        self.model.eval()

        self.real_features = []
        self.fake_features = []

    def collect_features(self, bbox, label, padding_mask, real=False,pku=False):
        if real and type(self.real_features) != list:
            return

        feats = self.model.extract_features(bbox.detach(), label, padding_mask,label_idx_replace_2=pku)
        features = self.real_features if real else self.fake_features
        features.append(feats.cpu().numpy())

    def compute_score(self):
        feats_1 = np.concatenate(self.fake_features)
        self.fake_features = []

        if type(self.real_features) == list:
            feats_2 = np.concatenate(self.real_features)
            self.real_features = feats_2
        else:
            feats_2 = self.real_features

        mu_1 = np.mean(feats_1, axis=0)
        sigma_1 = np.cov(feats_1, rowvar=False)
        mu_2 = np.mean(feats_2, axis=0)
        sigma_2 = np.cov(feats_2, rowvar=False)

        return calculate_frechet_distance(mu_1, sigma_1, mu_2, sigma_2)
    
class TransformerWithToken_layoutganpp(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers):
        super().__init__()

        self.token = nn.Parameter(torch.randn(1, 1, d_model))
        token_mask = torch.zeros(1, 1, dtype=torch.bool)
        self.register_buffer('token_mask', token_mask)

        self.core = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
            ), num_layers=num_layers)

    def forward(self, x, src_key_padding_mask):
        # x: [N, B, E]
        # padding_mask: [B, N]
        #   `False` for valid values
        #   `True` for padded values

        B = x.size(1)

        token = self.token.expand(-1, B, -1)
        x = torch.cat([token, x], dim=0)

        token_mask = self.token_mask.expand(B, -1)
        padding_mask = torch.cat([token_mask, src_key_padding_mask], dim=1)

        x = self.core(x, src_key_padding_mask=padding_mask)

        return x


class LayoutNet(nn.Module):
    def __init__(self, num_label):
        super().__init__()

        d_model = 256
        nhead = 4
        num_layers = 4
        max_bbox = 50

        # encoder
        self.emb_label = nn.Embedding(num_label, d_model)
        self.fc_bbox = nn.Linear(4, d_model)
        self.enc_fc_in = nn.Linear(d_model * 2, d_model)

        self.enc_transformer = TransformerWithToken_layoutganpp(d_model=d_model,
                                                                dim_feedforward=d_model // 2,
                                                                nhead=nhead, num_layers=num_layers)

        self.fc_out_disc = nn.Linear(d_model, 1)

        # decoder
        self.pos_token = nn.Parameter(torch.rand(max_bbox, 1, d_model))
        self.dec_fc_in = nn.Linear(d_model * 2, d_model)

        te = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                        dim_feedforward=d_model // 2)
        self.dec_transformer = nn.TransformerEncoder(te, num_layers=num_layers)

        self.fc_out_cls = nn.Linear(d_model, num_label)
        self.fc_out_bbox = nn.Linear(d_model, 4)

    def extract_features(self, bbox, label, padding_mask, label_idx_replace=False, label_idx_replace_2=False):
        b = self.fc_bbox(bbox)
        if label_idx_replace_2: # PKU
            label[label==2] = 3 # 'Logo' -> 'PICTOGRAM'
            label[label==3] = 444 # 'Underlay' -> 'BUTTON'
            label[label==444] = 4 # 'Underlay' -> 'BUTTON'
            label[label==1] = 2 # 'Text' -> 'TEXT'  
        else: #CGL
            label[label==1] = 3 # 'Logo' -> 'PICTOGRAM'
            label[label==4] = 3 # 'Embellishment' -> 'PICTOGRAM'
            label[label==3] = 444 # 'Underlay' -> 'BUTTON'
            label[label==5] = 2 # 'Highlighted text' -> 'TEXT'
            label[label==444] = 4 # 'Underlay' -> 'BUTTON'
            label[label==2] = 2 # 'Text' -> 'TEXT' 

        l = self.emb_label(label)
        x = self.enc_fc_in(torch.cat([b, l], dim=-1))
        x = torch.relu(x).permute(1, 0, 2)
        x = self.enc_transformer(x, padding_mask)
        return x[0]

    def forward(self, bbox, label, padding_mask):
        B, N, _ = bbox.size()
        x = self.extract_features(bbox, label, padding_mask)

        logit_disc = self.fc_out_disc(x).squeeze(-1)

        x = x.unsqueeze(0).expand(N, -1, -1)
        t = self.pos_token[:N].expand(-1, B, -1)
        x = torch.cat([x, t], dim=-1)
        x = torch.relu(self.dec_fc_in(x))

        x = self.dec_transformer(x, src_key_padding_mask=padding_mask)
        x = x.permute(1, 0, 2)[~padding_mask]

        # logit_cls: [M, L]    bbox_pred: [M, 4]
        logit_cls = self.fc_out_cls(x)
        bbox_pred = torch.sigmoid(self.fc_out_bbox(x))

        return logit_disc, logit_cls, bbox_pred

def remove_repeat(bbox, label):
    if bbox.size(0) == 0:
        return bbox, label 
    bbox_label = torch.cat((bbox, label.unsqueeze(1)), dim=1)  
    unique_bbox_label = []  
    for item in bbox_label:   
        same_bbox_label_exists = False  
        for unique_item in unique_bbox_label:  
            if torch.all(torch.eq(item, unique_item)):  
                same_bbox_label_exists = True  
                break  
        if not same_bbox_label_exists:  
            unique_bbox_label.append(item) 
     
    unique_bbox_label = torch.stack(unique_bbox_label) 
    unique_bbox = unique_bbox_label[:, :-1]  
    unique_label = unique_bbox_label[:, -1] 
    
    return unique_bbox, unique_label
def preprocess_batch(layouts, max_len: int):
    layout = defaultdict(list)
    empty_ids = []  # 0: empty 1: full
    for sample in layouts:
        if not isinstance(sample["bbox"], torch.Tensor):
            bbox, label = torch.tensor(sample["bbox"]), torch.tensor(sample["categories"])
        else:
            bbox, label = sample["bbox"], sample["categories"]
        bbox, label = remove_repeat(bbox, label)
        pad_len = max_len - label.size(0)

        if pad_len == max_len:
            empty_ids.append(0)
            pad_bbox = torch.tensor(np.full((max_len, 4), 0.0), dtype=torch.float)
            pad_label = torch.tensor(np.full((max_len,), 0), dtype=torch.long)
            mask = torch.tensor([False for _ in range(max_len)])
        else:
            empty_ids.append(1)  # not empty
            pad_bbox = torch.tensor(
                np.concatenate([bbox, np.full((pad_len, 4), 0.0)], axis=0),
                dtype=torch.float,
            )
            pad_label = torch.tensor(
                np.concatenate([label, np.full((pad_len,), 0)], axis=0),
                dtype=torch.long,
            )
            mask = torch.tensor(
                [True for _ in range(bbox.shape[0])] + [False for _ in range(pad_len)]
            ) 

        layout["bbox"].append(pad_bbox)
        layout["label"].append(pad_label)
        layout["mask"].append(mask)
        
    bbox = torch.stack(layout["bbox"], dim=0)
    label = torch.stack(layout["label"], dim=0)
    mask = torch.stack(layout["mask"], dim=0)
    
    padding_mask = ~mask.bool()  
    return bbox, label, padding_mask, mask, empty_ids 

def cal_layout_fid(model,box,box_gt,label,label_gt,batch_size=32,pku=False):
    W,H = 513,750
    filter_ids = []
    #box = [[ [b[0]/W,b[1]/H,b[2]/W,b[3]/H] for b in bb] for bb in box]
    box = [[ [(b[0]-b[2]/2)/W,(b[1]-b[3]/2)/H,(b[0]+b[2]/2)/W,(b[1]+b[3]/2)/H] for b in bb] for bb in box]
    #feats_gt= []
    #feats_gen = []

    k_generation = [{"bbox" : box[j], "categories" : label[j]} for j in range(len(box))]
    
    for i in range(0,len(k_generation),batch_size):
        i_end = min(i+batch_size,len(k_generation))
        batch = k_generation[i:i_end]
        max_len = max(len(g['categories']) for g in batch)
        if max_len == 0:  # prevent not generations
            max_len == 1    
        bbox, label, padding_mask, mask, empty_ids = preprocess_batch(batch, max_len)
        filter_ids.extend(empty_ids)
        
        mask_empty_ids = torch.tensor(empty_ids, dtype=torch.bool)
        bbox = torch.masked_select(bbox, mask_empty_ids.unsqueeze(1).unsqueeze(2)).reshape(-1, bbox.size(1), bbox.size(2)).contiguous()
        label = torch.masked_select(label, mask_empty_ids.unsqueeze(1)).reshape(-1, label.size(1)).contiguous()
        padding_mask = torch.masked_select(padding_mask, mask_empty_ids.unsqueeze(1)).reshape(-1, padding_mask.size(1)).contiguous()
        mask = torch.masked_select(mask, mask_empty_ids.unsqueeze(1)).reshape(-1, mask.size(1)).contiguous() 
        with torch.set_grad_enabled(False):
            model.collect_features(bbox, label, padding_mask,real=False,pku=pku)

    #box = [[ [b[0]/W,b[1]/H,b[2]/W,b[3]/H] for b in bb] for bb in box_gt]
    box = [[ [(b[0]-b[2]/2)/W,(b[1]-b[3]/2)/H,(b[0]+b[2]/2)/W,(b[1]+b[3]/2)/H] for b in bb] for bb in box_gt]
    
    filter_ids = []
    k_generation = [{"bbox" : box[j], "categories" : label_gt[j]} for j in range(len(box))]
    
    for i in range(0,len(k_generation),batch_size):
        i_end = min(i+batch_size,len(k_generation))
        batch = k_generation[i:i_end]
        max_len = max(len(g['categories']) for g in batch)
        if max_len == 0:  # prevent not generations
            max_len == 1    
        bbox, label, padding_mask, mask, empty_ids = preprocess_batch(batch, max_len)
        filter_ids.extend(empty_ids)
        
        mask_empty_ids = torch.tensor(empty_ids, dtype=torch.bool)
        bbox = torch.masked_select(bbox, mask_empty_ids.unsqueeze(1).unsqueeze(2)).reshape(-1, bbox.size(1), bbox.size(2)).contiguous()
        label = torch.masked_select(label, mask_empty_ids.unsqueeze(1)).reshape(-1, label.size(1)).contiguous()
        padding_mask = torch.masked_select(padding_mask, mask_empty_ids.unsqueeze(1)).reshape(-1, padding_mask.size(1)).contiguous()
        mask = torch.masked_select(mask, mask_empty_ids.unsqueeze(1)).reshape(-1, mask.size(1)).contiguous() 
        with torch.set_grad_enabled(False):
            model.collect_features(bbox, label, padding_mask,real=True,pku=pku)
    
    fid_score = model.compute_score()
    return fid_score