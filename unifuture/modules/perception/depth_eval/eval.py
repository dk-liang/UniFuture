from typing import List
from collections.abc import Callable
import numpy as np
import torch

from .alignment import align_depth_least_square


def eval_depth(
    pred: torch.Tensor, 
    target: torch.Tensor,
    metric_funcs: List[Callable],
    metric_tracker,
    clip_range=None,
    no_cuda=False,
):
    target_np = target.cpu().numpy()
    pred_np = pred.cpu().numpy()
    valid_mask = np.ones_like(pred_np, dtype=bool)
    depth_pred_np, scale, shift = align_depth_least_square(
        gt_arr=target_np,
        pred_arr=pred_np,
        valid_mask_arr=valid_mask,
        return_scale_shift=True,
    )

    # Clip to dataset min max
    if clip_range is not None:
        depth_pred_np = np.clip(
            depth_pred_np, a_min=clip_range[0], a_max=clip_range[1]
        )

    # clip to d > 0 for evaluation
    depth_pred_np = np.clip(depth_pred_np, a_min=1e-6, a_max=None)
    target = torch.clip(target, min=1e-6, max=None)

    # Evaluate (using CUDA if available)
    cuda_avail = torch.cuda.is_available() and not no_cuda
    device = torch.device("cuda" if cuda_avail else "cpu")
    sample_metric = []
    depth_pred_ts = torch.from_numpy(depth_pred_np).to(device)
    valid_mask_ts = torch.from_numpy(valid_mask).to(device)
    target = target.to(device)

    for met_func in metric_funcs:
        _metric_name = met_func.__name__
        _metric = met_func(depth_pred_ts, target, valid_mask_ts).item()
        sample_metric.append(_metric.__str__())
        metric_tracker.update(_metric_name, _metric)