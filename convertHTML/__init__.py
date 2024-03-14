# from dataset.rico import Rico
from convertHTML.publaynet import PubLayNet
from convertHTML.rico import Rico25Dataset
from convertHTML.magazine import MagezineDataset
from convertHTML.cgl import CGLDataset
from convertHTML.pku import PKUDataset


def get_dataset(name, datapath, split, transform=None, max_bbox_bins=32):
    if name == 'rico25':
        return Rico25Dataset(datapath, split, transform, max_bbox_bins=max_bbox_bins)

    elif name == 'publaynet':
        return PubLayNet(datapath, split, transform)#, max_bbox_bins=max_bbox_bins)

    elif name == 'magazine':
        return MagezineDataset(datapath, split, transform, max_bbox_bins=max_bbox_bins)
    
    elif name == "cgl":
        return CGLDataset(datapath,split,max_seq_length=25,transform=transform)

    elif name == "pku":
        return PKUDataset(datapath,split,max_seq_length=25,transform=transform)


    raise NotImplementedError(name)