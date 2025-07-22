import os
from PIL import Image
import copy

import numpy as np
from tqdm import tqdm
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from .meta import Metric, open_and_resize
from .meta import PAIRED_DATASET_SUBSET_FLAG as SUBSET_FLAG


def fid_update_from_path(paths, fid, real, max_sample=-1, batch_size=8, device="gpu:0", shuffle=True):
    root = paths["root"]
    length = len(paths["source"])
    if max_sample < 0:
        max_sample = length
    elif max_sample > length:
        print("Warning: max_sample ({}) is larger than the number of clips ({}). Changing to ({})... ".format(max_sample, length, length))
        max_sample = length
    
    indices = np.arange(length)
    if shuffle:
        np.random.shuffle(indices)

    for start in tqdm(range(0, max_sample, batch_size)):
        end = min(start + batch_size, max_sample)
        img_list = []
        for idx in indices[start: end]:
            frame = paths["source"][idx]
            if "paired_subset" in paths.keys():
                frame = frame.replace(SUBSET_FLAG, paths["paired_subset"])

            try:
                img = np.asarray(open_and_resize(os.path.join(root, frame)))      # (h, w, c)
            except:
                img = np.asarray(open_and_resize(os.path.join(paths["backup_root"], frame)))

            img = torch.from_numpy(copy.deepcopy(img)).permute(2, 0, 1).unsqueeze(0).to(device)        # (1, c, h, w)
            img_list.append(img)

        img_list = torch.cat(img_list, dim=0)                                           # (b, c, h, w)
        fid.update(img_list, real=real)


def fid_eval(paths_gt, paths_gen, feature=2048, max_sample=-1, batch_size=8, device="cuda:0", shuffle=True):
    assert feature in [64, 192, 768, 2048]
    fid = FID(feature=feature).to(device)
    fid_update_from_path(paths_gt, fid, real=True, max_sample=max_sample, batch_size=batch_size, device=device, shuffle=shuffle)
    fid_update_from_path(paths_gen, fid, real=False, max_sample=max_sample, batch_size=batch_size, device=device, shuffle=shuffle)
    return fid.compute().item()


class FID(Metric):
    def __init__(self, feature=2048, device="cuda:0", reset_real_features=False):
        super().__init__("FID")
        assert feature in [64, 192, 768, 2048]
        self.device = device
        self.reset_real_features = reset_real_features
        self.fid = FrechetInceptionDistance(feature=feature, reset_real_features=reset_real_features).to(device)

    def forward(self, gen_paths=None, max_sample=-1, batch_size=8, shuffle=True, **kwargs):
        fid_update_from_path(gen_paths, self.fid, real=False, max_sample=max_sample, batch_size=batch_size, device=self.device, shuffle=shuffle)
        fid = self.fid.compute()
        print('FID: {}'.format(fid))
        return fid

    def update_gt(self, gt_paths=None, max_sample=-1, batch_size=8, shuffle=True, **kwargs):
        fid_update_from_path(gt_paths, self.fid, real=True, max_sample=max_sample, batch_size=batch_size, device=self.device, shuffle=shuffle)