import os
import math
from PIL import Image

import numpy as np
from tqdm import tqdm
import torch

from .fvd_utils import get_fvd_logits, frechet_distance, polynomial_mmd
from .pytorch_i3d import InceptionI3d
from .meta import I3D_PATH, MIN_I3D_TIME, Metric, open_and_resize
from .meta import PAIRED_DATASET_SUBSET_FLAG as SUBSET_FLAG

PRINT_WARNING = False


def i3d_process_img_paths(paths, i3d, freq, max_sample=-1, batch_size=8, device='cuda:0', shuffle=True, mode='repeat'):
    global PRINT_WARNING
    root = paths["root"]
    length = len(paths["source"])
    gen_startid = paths["gen_startid"]
    if max_sample < 0:
        max_sample = length
    elif max_sample > length:
        print("Warning: max_sample ({}) is larger than the number of clips ({}). Changing to ({})... ".format(max_sample, length, length))
        max_sample = length
    indices = np.arange(length)
    if shuffle:
        np.random.shuffle(indices)
    embs = []
    for idx in tqdm(indices[:max_sample]):
        datum = paths["source"][idx]
        videos = []
        for frame in datum[gen_startid:]:
            if "paired_subset" in paths.keys():
                frame = frame.replace(SUBSET_FLAG, paths["paired_subset"])

            try:
                img = np.asarray(open_and_resize(os.path.join(root, frame)))     # (h, w, c)
            except:
                img = np.asarray(open_and_resize(os.path.join(paths["backup_root"], frame)))

            for _ in range(math.ceil(freq / paths["freq"])):
                videos.append(img)
        # print(len(videos))
        for _ in range(len(videos), MIN_I3D_TIME):
            assert False, "The length of video is less than MIN_I3D_TIME."
            # if mode == 'zero':
            #     videos.append(np.zeros_like(img))
            #     if not PRINT_WARNING:
            #         print("Warning: the length of video is less than MIN_I3D_TIME. Padding with zeros.")
            #         PRINT_WARNING = True
            # elif mode == 'repeat':  
            #     videos.append(videos[-1])
            #     if not PRINT_WARNING:
            #         print("Warning: the length of video is less than MIN_I3D_TIME. Padding with the last frame.")
            #         PRINT_WARNING = True
        videos = np.expand_dims(np.stack(videos, axis=0), 0)            # (1, t, h, w, c)
        # print(idx, ": ", videos.shape, end=" -> ")
        emb = get_fvd_logits(videos, i3d=i3d, device=device, batch_size=batch_size)
        # print(emb.shape)
        embs.append(emb)

    embs = torch.cat(embs, 0)
    return embs


def load_fvd_model(device):
    i3d = InceptionI3d(400, in_channels=3).to(device)
    i3d.load_state_dict(torch.load(I3D_PATH, map_location=device))
    i3d.eval()
    return i3d


def fvd_kvd_eval_single(gt_paths, gen_paths, i3d, max_sample=-1, batch_size=8, device='cuda:0', shuffle=True, self_compare=False):
    freq = max(gt_paths["freq"], gen_paths["freq"])
    embs_gt = i3d_process_img_paths(gt_paths, i3d, freq, max_sample=max_sample, batch_size=batch_size, device=device, shuffle=shuffle) \
                if isinstance(gt_paths, dict) else gt_paths
    if self_compare:
        embs_gen = embs_gt
    else:
        embs_gen = i3d_process_img_paths(gen_paths, i3d, freq, max_sample=max_sample, batch_size=batch_size, device=device, shuffle=shuffle) \
                    if isinstance(gen_paths, dict) else gen_paths
    fvd = frechet_distance(embs_gen, embs_gt)
    kvd = polynomial_mmd(embs_gen.cpu(), embs_gt.cpu())
    return fvd, kvd
    

def fvd_kvd_eval_multi(gt_paths, gen_paths, max_sample=-1, batch_size=8, device='cuda:0', shuffle=True, n_runs=1, self_compare=False):
    print('Loading i3d...')
    i3d = load_fvd_model(device)

    fvd_list = []
    kvd_list = []
    for i in range(n_runs):
        print(f'Run {i+1}/{n_runs}')
        fvd, kvd = fvd_kvd_eval_single(gt_paths, gen_paths, i3d, max_sample=max_sample, batch_size=batch_size, device=device,
                                       shuffle=shuffle, self_compare=self_compare)
        fvd_list.append(fvd.item())
        kvd_list.append(kvd.item())
        print(f'FVD: {fvd.item():.4f}, KVD: {kvd.item():.4f}')

    return fvd_list, kvd_list



class FVD_KVD(Metric):
    def __init__(self, device):
        super().__init__("FVD_KVD")
        self.i3d = load_fvd_model(device)
        self.device = device
        self.gt_paths = None

    def forward(self, gen_paths=None, max_sample=-1, batch_size=8, shuffle=True, n_runs=1, self_compare=False, fix_gt=True, freq=-1, **kwargs):
        if (self.gt_paths is None) or (not fix_gt):
            self.update_gt(**kwargs)
        
        if freq < 0:
            self.freq = max(self.freq, gen_paths["freq"])
        else:
            self.freq = freq

        embs_gen = i3d_process_img_paths(gen_paths, self.i3d, freq, max_sample=max_sample, batch_size=batch_size, device=self.device, shuffle=shuffle) \
                    if isinstance(gen_paths, dict) else gen_paths

        fvd_list = []
        kvd_list = []
        for i in range(n_runs):
            print(f'Run {i+1}/{n_runs}')
            fvd = frechet_distance(embs_gen, self.embs_gt)
            # kvd = polynomial_mmd(embs_gen.cpu(), self.embs_gt.cpu())
            fvd_list.append(fvd.item())
            # kvd_list.append(kvd.item())
            print(f'FVD: {fvd.item():.4f}') #, KVD: {kvd.item():.4f}')
        
        # print(fvd_list)
        # print(kvd_list)
        # fvd_mean, kvd_mean = np.mean(fvd_list).item(), np.mean(kvd_list).item()
        fvd_mean = np.mean(fvd_list).item()
        print('FVD: {}'.format(fvd_mean))
        # print('KVD: {}'.format(kvd_mean))
        # return fvd_mean, kvd_mean
        return fvd_mean

    def update_gt(self, gt_paths=None, max_sample=-1, batch_size=8, shuffle=True, n_runs=1, self_compare=False, fix_gt=True, freq=-1, **kwargs):
        self.gt_paths = gt_paths
        self.freq = freq if freq > 0 else gt_paths["freq"]
        self.embs_gt = i3d_process_img_paths(gt_paths, self.i3d, freq, max_sample=max_sample, batch_size=batch_size, device=self.device, shuffle=shuffle) \
                if isinstance(gt_paths, dict) else gt_paths
        print("GT path updated.")