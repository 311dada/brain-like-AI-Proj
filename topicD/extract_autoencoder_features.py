import torch
import torch.utils.data as data
import os 
import sys
import h5py
import fire
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
sys.path.append("/mnt/lustre/sjtu/home/xnx98/utils/")
import kaldi_io

from models import *
from datasets.dataset import create_dataloader



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
def build_phone_dict(phone_path):
    phonedict = {}
    for line in open(phone_path):
        phone, idx = line.strip().split()
        phonedict[int(idx)] = phone
    return phonedict


np.random.seed(1)
        
def extract(model_path, tag, write_hdf5, batch_size=2048):
    saved = torch.load(model_path, map_location=lambda storage, loc: storage)
    if isinstance(saved["model"], dict):
        encoder = saved["model"]["encoder"]
    else:
        encoder = saved["model"].encoder
    config_params = saved["config"]
    context = config_params["context"]
    
    encoder = encoder.to(device)

    scaler = torch.load(os.path.join(os.path.dirname(model_path), config_params["scaler_path"]))

    phone_path = "/mnt/lustre/sjtu/home/zkz01/timit/data_for_brainlike/phones.txt"
    idx2phone = build_phone_dict(phone_path)
    dataloader = create_dataloader(config_params["test_feat_stream"].format(context, context), 
                                   config_params["test_label_stream"].format(context, context),
                                   transform=scaler.transform,
                                   shuffle=False,
                                   batch_size=batch_size)
    # writer = SummaryWriter()

    phones = ["sil", "cl", "aw", "ao"]

    with torch.no_grad(), tqdm(total=len(dataloader)) as pbar, h5py.File(write_hdf5, "w") as store:
        for batch_idx, (feats, labels, keys) in enumerate(dataloader):
            feats = feats.to(device)
            encoder_feats = encoder(feats).cpu().numpy()
            
            # feats_write = []
            # labels_write = []
            for idx, label in enumerate(labels.tolist()):
                if idx2phone[label] in phones:
                    store["{}_{}/feature".format(keys[0][idx], keys[1][idx].item())] = encoder_feats[idx]
                    store["{}_{}/label".format(keys[0][idx], keys[1][idx].item())] = idx2phone[label]
                    # if label in phones:
                        # feats_write.append(encoder_feats[idx])
                        # labels_write.append(label)


                # writer.add_embedding(np.stack(feats_write),
                                     # np.array(labels_write),
                                     # global_step=batch_idx,
                                     # tag=tag)
            pbar.update()


if __name__ == "__main__":
    fire.Fire(extract)

