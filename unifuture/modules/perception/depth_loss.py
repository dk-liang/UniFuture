import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from unifuture.modules.diffusionmodules.util import fourier_filter

# losses below for depth branch loss
def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero() 

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


def structure_preserving_loss(pred, tgt):
    # structural loss
    pred_hf = fourier_filter(pred, scale=0.)
    tgt_hf = fourier_filter(tgt, scale=0.)
    depth_hf_loss = torch.mean(((pred_hf - tgt_hf).abs()).reshape(tgt.shape[0], -1), 1).mean()
    return depth_hf_loss


def dynamic_enhancement_loss(pred, tgt, num_frames):
    # dynamic enhancement loss
    pred_seq = rearrange(pred, "(b t) ... -> b t ...", t=num_frames)
    tgt_seq = rearrange(tgt, "(b t) ... -> b t ...", t=num_frames)
    bs = tgt.shape[0] // num_frames
    depth_aux_weight = ((tgt_seq[:, 1:] - tgt_seq[:, :-1]) - (pred_seq[:, 1:] - pred_seq[:, :-1])) ** 2
    tmp_h, tmp_w = depth_aux_weight.shape[-2], depth_aux_weight.shape[-1]
    depth_aux_weight = rearrange(depth_aux_weight, "b t c h w -> b (t h w) c", c=4)
    depth_aux_w = F.normalize(depth_aux_weight, p=2)
    depth_aux_w = rearrange(depth_aux_w, "b (t h w) c -> b t c h w", t=num_frames - 1, h=tmp_h, w=tmp_w)
    depth_aux_w = torch.cat((torch.zeros(bs, 1, *depth_aux_w.shape[2:]).to(depth_aux_w), depth_aux_w), dim=1)
    depth_aux_w = rearrange(depth_aux_w, "b t ... -> (b t) ...").reshape(tgt.shape[0], -1)
    depth_dynamic_loss = torch.mean(
        F.mse_loss(pred, tgt, reduction='none').reshape(tgt.shape[0], -1) * depth_aux_w.detach(), dim=1
    ).mean()

    return depth_dynamic_loss


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        if len(prediction.shape) == 4:
            prediction = prediction.squeeze(1)
        if len(target.shape) == 4:
            target = target.squeeze(1)
        if len(mask.shape) == 4:
            mask = mask.squeeze(1)

        scale, shift = compute_scale_and_shift(prediction, target, mask)  
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


# below are loss weight schedulars
class LinearDepthLossWeightScheduler(nn.Module):
    def __init__(self, start_weight, end_weight, start_step, end_step):
        super().__init__()

        assert start_step < end_step
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.start_step = start_step
        self.end_step = end_step

        self.slope = (end_weight - start_weight) / (end_step - start_step)

    def forward(self, step):
        weight = self.slope * (step - self.start_step) + self.start_weight
        weight = min(max(weight, self.start_weight), self.end_weight)
        return weight
