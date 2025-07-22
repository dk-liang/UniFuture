from abc import abstractmethod
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import cv2

from unifuture.util import instantiate_from_config


def split_tensor_list(tensor_list, split_size=1, dim=0, uc_mode=False):
    '''split tensors, which is given in the list form

    Args:
        tensor_list (List[torch.Tensor]): the list of tensors, the len of list is N, and each entry is a tensor
        split_size (int, optional): the split size, default to 1
        dim (int, optional): the split dim, default to 0,
        have_eq_size (bool): whether the input tensors have the same shape
        uc_mode (bool): whether the tensors have contional/unconditional information together

    Outputs:
        splited_tensor_list (List[List[torch.Tensor]]): the list of splited tensors, and the len of list is the number of chunks
            (e.g., default to B / 1), and each entry contains N tensors, of shape [split_size] x C x L by default
    '''
    def base_split_op(base_tensors):
        outputs = list()
        splited_tensor_nested_list = list()
        for tensor in base_tensors:
            splited_tensor_nested_list.append(torch.split(tensor, split_size_or_sections=split_size, dim=dim))
        
        for chunks in zip(*splited_tensor_nested_list):
            outputs.append(chunks)
        return outputs

    if not uc_mode:
        return base_split_op(tensor_list)

    else:
        tensor_list_u = [hs.chunk(2)[0] for hs in tensor_list]
        tensor_list_c = [hs.chunk(2)[1] for hs in tensor_list]

        splited_tensor_nested_list_u = base_split_op(tensor_list_u)
        splited_tensor_nested_list_c = base_split_op(tensor_list_c)

        outputs = list()
        for splited_tensor_list_u, splited_tensor_list_c in zip(splited_tensor_nested_list_u, splited_tensor_nested_list_c):
            outputs.append([
                torch.cat([tensor_u, tensor_c], dim=0) for tensor_u, tensor_c in zip(splited_tensor_list_u, splited_tensor_list_c)
            ])

        return outputs


def get_depth_vis(img: torch.Tensor, depth: torch.Tensor, to_numpy=False):
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    vis_list = list()
    for sample_id in range(depth.shape[0]):
        d_tgt = depth[sample_id, 0].cpu().numpy()
        d_tgt = (d_tgt - d_tgt.min()) / (d_tgt.max() - d_tgt.min()) * 255.0
        d_tgt = d_tgt.astype(np.uint8)
        d_tgt = (cmap(d_tgt)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        vis_res = d_tgt
        if img is not None:
            raw_img = img[sample_id].cpu().numpy()
            raw_img = np.transpose(raw_img, [1, 2, 0])
            raw_img = (raw_img + 1.0) / 2.0 * 255.0
            raw_img = raw_img.astype(np.uint8)
            split_region = np.ones((raw_img.shape[0], 50, 3), dtype=np.uint8) * 255
            vis_res = cv2.hconcat([raw_img, split_region, d_tgt])

        if to_numpy:
            vis_list.append(vis_res)
        else:
            img_preprocessor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 2.0 - 1.0)
            ])
            vis_list.append(img_preprocessor(vis_res))

    return vis_list


# borrowed from MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
def get_2d_sincos_pos_embed(embed_dim, grid_height, grid_width, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_height, dtype=np.float32)
    grid_w = np.arange(grid_width, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_height, grid_width])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class SineCosinePositonalEncoding(nn.Module):
    '''borrowed from MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py'''
    def __init__(
        self,
        grid_height, 
        grid_width,
        num_features,
        *args,
        **kwargs
    ):
        super().__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_features = num_features
        self.pos_embed = nn.Parameter(torch.zeros(1, num_features, grid_height, grid_width), requires_grad=False)  # fixed sin-cos embedding
        self.initialize_weights()
    
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.num_features, self.grid_height, self.grid_width, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().view(self.grid_height, self.grid_width, -1).permute(2, 0, 1).unsqueeze(0))

    def forward(self, x, *args, **kwargs):
        return x + self.pos_embed


class LearnablePositionalEmbedding(nn.Module):
    def __init__(
        self,
        grid_height,
        grid_width,
        num_features,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_features = num_features
        self.pos_embed = nn.Parameter(torch.zeros(1, num_features, grid_height, grid_width), requires_grad=True)  # learnable embedding

    def forward(self, x, *args, **kwargs):
        return x + self.pos_embed


class MultiscalePositionalEmbeddingWrapper(nn.Module):
    def __init__(
        self,
        base_target,
        base_num_features,
        base_grid_height,
        base_grid_width,
        channel_mults,
        down_scales,
        *args, 
        **kwargs,
    ):
        super().__init__()
        assert len(down_scales) == len(channel_mults)

        self.pe_modules = nn.ModuleList()
        for lvl_id, (ch_mult, scale) in enumerate(zip(channel_mults, down_scales)):
            pe_config = dict(
                target=base_target,
                params=dict(
                    grid_height = base_grid_height // scale,
                    grid_width = base_grid_width // scale,
                    num_features = base_num_features * ch_mult
                )
            )
            self.pe_modules.append(instantiate_from_config(pe_config))

    def forward(self, x, level_id, *args, **kwargs):
        pe_module = self.pe_modules[level_id]
        return pe_module(x)


class DepthTargetMappingBase(object):
    def __init__(self, near_plane, far_plane):
        self.near_plane = near_plane
        self.far_plane = far_plane

    @abstractmethod
    def map(self, depth, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def inverse_map(self, depth_mapped, *args, **kwargs):
        raise NotImplementedError


class MinMaxDepthTargetMapping(DepthTargetMappingBase):
    def __init__(self, near_plane=-1.0, far_plane=1.0, per_frame_norm=False):
        super().__init__(near_plane, far_plane)
        self.depth_range = far_plane - near_plane
        self.per_frame_norm = per_frame_norm

    def map(self, depth, *args, **kwargs):
        # depth: T x 1 x H x W
        if self.per_frame_norm:
            _min = torch.amin(depth, dim=[2, 3], keepdim=True)
            _max = torch.amax(depth, dim=[2, 3], keepdim=True)
        else:
            _min, _max = depth.min(), depth.max()
        depth_standard = (depth - _min) / (_max - _min)  # 0~1
        depth_mapped = depth_standard * self.depth_range + self.near_plane
        return depth_mapped, _min, _max
    
    def inverse_map(self, depth_mapped, ori_depth_min, ori_depth_max, *args, **kwargs):
        depth_standard = (depth_mapped - self.near_plane) / self.depth_range
        depth = depth_standard * (ori_depth_max - ori_depth_min) + ori_depth_min
        return depth


class QuantileDepthTargetMapping(DepthTargetMappingBase):
    def __init__(self, near_plane=-1.0, far_plane=1.0, min_quantile=0.02, clamp=True, per_frame_norm=False):
        super().__init__(near_plane=near_plane, far_plane=far_plane)
        self.depth_range = far_plane - near_plane

        self.min_quantile = min_quantile
        self.max_quantile = 1 - min_quantile
        self.clamp = clamp
        self.per_frame_norm = per_frame_norm

    def map(self, depth, *args, **kwargs):
        '''
        Args:
            depth (torch.Tensor): B x 1 x H x W
        '''
        depth = depth.float()
        if self.per_frame_norm:
            _min, _max = torch.quantile(
                depth.flatten(2, 3), 
                torch.tensor([self.min_quantile, self.max_quantile], device=depth.device), 
                dim=-1, 
                keepdim=True
            )
            _min = _min.view(-1, 1, 1, 1)
            _max = _max.view(-1, 1, 1, 1)
        else:
            _min, _max = torch.quantile(depth, torch.tensor([self.min_quantile, self.max_quantile], device=depth.device))
        depth_standard = (depth - _min) / (_max - _min)
        depth_mapped = depth_standard * self.depth_range + self.near_plane
        if self.clamp:
            depth_mapped = torch.clamp(depth_mapped, min=self.near_plane, max=self.far_plane)
        return depth_mapped, _min, _max
    
    def inverse_map(self, depth_mapped, ori_depth_min, ori_depth_max, *args, **kwargs):
        depth_standard = (depth_mapped - self.near_plane) / self.depth_range
        depth = depth_standard * (ori_depth_max - ori_depth_min) + ori_depth_min
        return depth
