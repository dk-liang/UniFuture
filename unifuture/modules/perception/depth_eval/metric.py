# NOTE: we borrowed this file from Marigold https://github.com/prs-eth/Marigold/blob/main/src/util/metric.py

import pandas as pd
import torch


# Adapted from: https://github.com/victoresque/pytorch-template/blob/master/utils/util.py

class DepthMetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

def get_mask_within_thresh(output, target, is_intervel=True, thresh=1000):
    if is_intervel:
        actual_output = 1. / output
        actual_target = 1. / target
    within_mask = actual_target < thresh
    actual_output[~within_mask] = 0
    h_th = torch.quantile(actual_output, 0.98)
    l_th = torch.quantile(actual_output, 0.02)
    within_mask = within_mask & ((actual_output > l_th) & (actual_output < h_th))
    return within_mask, actual_output, actual_target

def abs_relative_difference(output, target, valid_mask=None, thresh=1000):
    within_mask, actual_output, actual_target = get_mask_within_thresh(output, target, is_intervel=True, thresh=thresh)
    abs_relative_diff = torch.abs(actual_output - actual_target) / actual_target 
    if valid_mask is not None:
        within_mask = valid_mask * within_mask
    abs_relative_diff[~within_mask] = 0
    n = within_mask.sum((-1, -2))
    abs_relative_diff = torch.sum(abs_relative_diff, (-1, -2)) / n
    return abs_relative_diff.mean()

def squared_relative_difference(output, target, valid_mask=None, thresh=1000):
    within_mask, actual_output, actual_target = get_mask_within_thresh(output, target, is_intervel=True, thresh=thresh)
    square_relative_diff = (
        torch.pow(torch.abs(actual_output - actual_target), 2) / actual_output
    )
    if valid_mask is not None:
        within_mask = valid_mask * within_mask
    square_relative_diff[~within_mask] = 0
    n = within_mask.sum((-1, -2))
    square_relative_diff = torch.sum(square_relative_diff, (-1, -2)) / n
    return square_relative_diff.mean()

def rmse_linear(output, target, valid_mask=None, thresh=1000):
    within_mask, actual_output, actual_target = get_mask_within_thresh(output, target, is_intervel=True, thresh=thresh)
    diff = actual_output - actual_target
    if valid_mask is not None:
        within_mask = valid_mask * within_mask
    diff[~within_mask] = 0
    n = within_mask.sum((-1, -2))
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n
    rmse = torch.sqrt(mse)
    return rmse.mean()

def rmse_log(output, target, valid_mask=None, thresh=1000):
    within_mask, actual_output, actual_target = get_mask_within_thresh(output, target, is_intervel=True, thresh=thresh)
    diff = torch.log(actual_output) - torch.log(actual_target)
    if valid_mask is not None:
        within_mask = valid_mask * within_mask
    diff[~within_mask] = 0
    n = within_mask.sum((-1, -2))
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n  # [B]
    rmse = torch.sqrt(mse)
    return rmse.mean()


def log10(output, target, valid_mask=None, thresh=1000):
    within_mask, actual_output, actual_target = get_mask_within_thresh(output, target, is_intervel=True, thresh=thresh)
    diff = torch.abs(torch.log10(actual_output) - torch.log10(actual_target))
    if valid_mask is not None:
        within_mask = valid_mask * within_mask
    diff[~within_mask] = 0
    n = within_mask.sum((-1, -2))
    diff = torch.sum(diff, (-1, -2)) / n
    return diff.mean()


# adapt from: https://github.com/imran3180/depth-map-prediction/blob/master/main.py
def threshold_percentage(output, target, threshold_val, valid_mask=None, mask_thresh=1000):
    within_mask, actual_output, actual_target = get_mask_within_thresh(output, target, is_intervel=True, thresh=mask_thresh)
    d1 = actual_output / actual_target
    d2 = actual_target / actual_output
    max_d1_d2 = torch.max(d1, d2)
    zero = torch.zeros(*output.shape)
    one = torch.ones(*output.shape)
    bit_mat = torch.where(max_d1_d2.cpu() < threshold_val, one, zero)
    if valid_mask is not None:
        within_mask = valid_mask * within_mask
    bit_mat[~within_mask] = 0
    n = within_mask.sum((-1, -2))
    count_mat = torch.sum(bit_mat, (-1, -2))
    threshold_mat = count_mat / n.cpu()
    return threshold_mat.mean()


def delta1_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25, valid_mask)


def delta2_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**2, valid_mask)


def delta3_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**3, valid_mask)


def i_rmse(output, target, valid_mask=None, thresh=1000):
    within_mask, actual_output, actual_target = get_mask_within_thresh(output, target, is_intervel=True, thresh=thresh)

    diff = actual_output - actual_target
    if valid_mask is not None:
        within_mask = valid_mask * within_mask
    diff[~within_mask] = 0
    n = within_mask.sum((-1, -2))
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n
    rmse = torch.sqrt(mse)
    return rmse.mean()


def silog_rmse(depth_pred, depth_gt, valid_mask=None, thresh=1000):
    within_mask, depth_pred, depth_gt = get_mask_within_thresh(depth_pred, depth_gt, is_intervel=True, thresh=thresh)
    diff = torch.log(depth_pred) - torch.log(depth_gt)
    if valid_mask is not None:
        within_mask = valid_mask * within_mask
    diff[~within_mask] = 0
    n = within_mask.sum((-1, -2))
    diff2 = torch.pow(diff, 2)
    first_term = torch.sum(diff2, (-1, -2)) / n
    second_term = torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
    loss = torch.sqrt(torch.mean(first_term - second_term)) * 100
    return loss
