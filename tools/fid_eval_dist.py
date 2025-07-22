import argparse
import torch
import os
import sys
import os.path as osp

sys.path.append(os.getcwd())
from unifuture.modules.perception.depth_utils import get_depth_vis
from unifuture.fvd_utils.fid_utils import FlexibleFrechetInceptionDistance

def parse_args(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--feats_dir",
        type=str,
        required=True
    )
    return parser


def load_and_concat_from_dir(feats_dir, num_files=8, mode='fid'):
    real_embeddings = list()
    fake_embeddings = list()
    
    for id in range(num_files):
        file_path = osp.join(feats_dir, f'cuda_{id}_{mode}_feats.pt')
        if osp.exists(file_path):
            print(f'Now loading feats in {file_path}')
            tensor_dict = torch.load(file_path)
            real_embeddings.append(tensor_dict['real_feats'])
            fake_embeddings.append(tensor_dict['fake_feats'])
        else:
            print(f'Skip loading {file_path}')

    return torch.cat(real_embeddings, dim=0), torch.cat(fake_embeddings, dim=0)


def calc_metric(real_embeddings, fake_embeddings, mode='fid'):
    print(f'Now calculating {mode.upper()}, total sample num: {real_embeddings.shape[0]}')
    fid = FlexibleFrechetInceptionDistance()
    fid.update(real_embeddings, real=True, skip_feat_extraction=True)
    fid.update(fake_embeddings, real=False, skip_feat_extraction=True)
    return fid.compute()


if __name__ == '__main__':
    opt = parse_args().parse_args()
    fid_real_feats, fid_fake_feats = load_and_concat_from_dir(opt.feats_dir, mode='fid')
    fid = calc_metric(fid_real_feats, fid_fake_feats, mode='fid')
    print(f'Total sample num: {fid_real_feats.shape[0]}, FID: {fid}')
