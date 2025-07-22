import random
from typing import Dict, List, Union
import os.path as osp

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from unifuture.modules.diffusionmodules.util import fourier_filter
from unifuture.modules.encoders.modules import GeneralConditioner
from unifuture.modules.perception.depth_utils import split_tensor_list, get_depth_vis
from unifuture.modules.perception.depth_loss import ScaleAndShiftInvariantLoss, structure_preserving_loss, dynamic_enhancement_loss
from unifuture.util import append_dims, instantiate_from_config
from .denoiser import Denoiser


class StandardDiffusionLoss(nn.Module):
    SUPPORTED_PERCEPTION_LOSS_MODE = ('latent', 'pixel', 'latent_pixel', 'pixel_latent')

    def __init__(
            self,
            sigma_sampler_config: dict,
            loss_weighting_config: dict,
            loss_type: str = "l2",
            use_additional_loss: bool = False,
            offset_noise_level: float = 0.0,
            additional_loss_weight: float = 0.0,
            num_frames: int = 25,
            replace_cond_frames: bool = False,
            cond_frames_choices: Union[List, None] = None,
            # for perception branches
            perception_loss_mode: str = 'latent',
            perception_loss_weight_schedulers: dict = None,
            noise_level_perception_loss_weight: bool = False,
            depth_loss_weight: Union[float, dict] = 0.0,
            depth_ssi_loss_config: dict = None,
            use_additional_loss_for_depth: bool = False,
            additional_loss_weight_depth: float = 0.0,
            mutual_perception_loss: bool = False,  # whether to calculate losses between predicted images and perception results
            max_num_frame_for_decoding: int = -1,  # prevent OOM when perception_loss_mode is 'latent_pixel'
    ):
        super().__init__()
        assert loss_type in ["l2", "l1"]
        self.loss_type = loss_type
        self.use_additional_loss = use_additional_loss

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.loss_weighting = instantiate_from_config(loss_weighting_config)

        self.offset_noise_level = offset_noise_level
        self.additional_loss_weight = additional_loss_weight
        self.num_frames = num_frames
        self.replace_cond_frames = replace_cond_frames
        self.cond_frames_choices = cond_frames_choices

        ################################### for perception branches ###################################
        self.perception_loss_mode = perception_loss_mode

        # build loss weight schedulars
        self.perception_loss_weight_schedulers = None
        if perception_loss_weight_schedulers is not None:
            self.perception_loss_weight_schedulers = dict()
            for key, schedular_cfg in perception_loss_weight_schedulers.items():
                self.perception_loss_weight_schedulers[key] = instantiate_from_config(schedular_cfg)

        self.noise_level_perception_loss_weight = noise_level_perception_loss_weight
        self.depth_ssi_loss = ScaleAndShiftInvariantLoss(**depth_ssi_loss_config) if depth_ssi_loss_config else None 
        if isinstance(depth_loss_weight, float) or isinstance(depth_loss_weight, int):
            self.depth_loss_weight = depth_loss_weight
        else:
            self.depth_loss_weight = dict(depth_loss_weight)
        self.use_additional_loss_for_depth = use_additional_loss_for_depth
        self.additional_loss_weight_depth = additional_loss_weight_depth
        self.mutual_perception_loss = mutual_perception_loss
        assert self.perception_loss_mode in self.SUPPORTED_PERCEPTION_LOSS_MODE, f'Only support {self.SUPPORTED_PERCEPTION_LOSS_MODE}, got: {perception_loss_mode}'
        self.max_num_frame_for_decoding = max_num_frame_for_decoding

    def get_noised_input(
            self,
            sigmas_bc: torch.Tensor,
            noise: torch.Tensor,
            input: torch.Tensor
    ) -> torch.Tensor:
        noised_input = input + noise * sigmas_bc
        return noised_input

    def forward(
            self,
            network: nn.Module,
            denoiser: Denoiser,
            conditioner: GeneralConditioner,
            input: torch.Tensor,
            batch: Dict,
            depth_branch: nn.Module = None,
            depth_target: torch.Tensor = None,
            *args,
            **kwargs,
    ) -> torch.Tensor:
        cond = conditioner(batch)
        return self._forward(network, denoiser, cond, input, depth_branch, depth_target, *args, **kwargs)

    def _forward(
            self,
            network: nn.Module,
            denoiser: Denoiser,
            cond: Dict,
            input: torch.Tensor,
            depth_target: torch.Tensor = None,
            *args,
            **kwargs
    ):
        sigmas = self.sigma_sampler(input.shape[0]).to(input)
        cond_mask = torch.zeros_like(sigmas)
        if self.replace_cond_frames:
            cond_mask = rearrange(cond_mask, "(b t) -> b t", t=self.num_frames)
            for each_cond_mask in cond_mask:
                assert len(self.cond_frames_choices[-1]) < self.num_frames
                weights = [2 ** n for n in range(len(self.cond_frames_choices))]
                cond_indices = random.choices(self.cond_frames_choices, weights=weights, k=1)[0]
                if cond_indices:
                    each_cond_mask[cond_indices] = 1
            cond_mask = rearrange(cond_mask, "b t -> (b t)")
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:  # the entire channel is shifted together
            offset_shape = (input.shape[0], input.shape[1])
            rand_init = torch.randn(offset_shape, device=input.device)
            noise = noise + self.offset_noise_level * append_dims(rand_init, input.ndim)
        if self.replace_cond_frames:
            sigmas_bc = append_dims((1 - cond_mask) * sigmas, input.ndim)
        else:
            sigmas_bc = append_dims(sigmas, input.ndim)
        noised_input = self.get_noised_input(sigmas_bc, noise, input)

        network.diffusion_model.set_save_intermediate(flag=True)  # save intermediate features
        model_output, depth_latent = denoiser(network, noised_input, sigmas, cond, cond_mask)  # denoised latent
        network.diffusion_model.set_save_intermediate(flag=False) 
        w = append_dims(self.loss_weighting(sigmas), input.ndim)

        if self.replace_cond_frames:  # ignore mask predictions
            predict = model_output * append_dims(1 - cond_mask, input.ndim) + input * append_dims(cond_mask, input.ndim)
        else:
            predict = model_output

        ############################ Perception branches ###########################
        perception_mode = kwargs.get('perception_mode', 'extra_dec')
        perception_feedback = kwargs.get('perception_feedback', False)
        feedback_layer = kwargs.get('perception_feedback_layer', None)

        auxiliary_predict = dict()
        auxiliary_target = dict()

        # 1. depth branch
        if depth_latent is not None:
            depth_predict_dict = dict(latent=depth_latent)
            depth_target_dict = dict(latent=depth_target)

            if perception_mode.startswith('align_latent') and self.perception_loss_mode == 'latent_pixel':
                dec_func = kwargs.get('dec_func', None)
                depth_target_pixel = kwargs.get('depth_target_pixel', None)
                assert dec_func is not None and depth_target_pixel is not None

                if self.max_num_frame_for_decoding > 0:
                    # here we have to sample part of depth_latent to sent into AE decoder, for saving momery
                    sampled_idx = torch.randint(
                        low=0, high=depth_latent.shape[0], 
                        size=[self.max_num_frame_for_decoding],
                        dtype=torch.long,
                        device=depth_latent.device
                    )
                    depth_latent = torch.index_select(depth_latent, dim=0, index=sampled_idx)
                    depth_target_pixel = torch.index_select(depth_target_pixel, dim=0, index=sampled_idx)

                decoded_depth = dec_func(depth_latent)
                if decoded_depth.shape[1] == 3:
                    decoded_depth = torch.mean(decoded_depth, dim=1, keepdim=True)
                depth_predict_dict['pixel'] = decoded_depth 
                depth_target_dict['pixel'] = depth_target_pixel

            if 'latent' in perception_mode and perception_feedback:
                assert feedback_layer is not None
                depth_feedback = depth_predict_dict['latent']
                feedback = feedback_layer(depth_feedback)
                predict += feedback

            if self.mutual_perception_loss:  # will use predicted videos to calculate depth target as extra regularization
                dec_func = kwargs.get('dec_func')
                depth_target_func = kwargs.get('depth_target_func')

                with torch.no_grad():
                    # 1. transform predicted latent to RGB
                    decoded_predict = dec_func(predict)
                    # 2. obtain the depth target
                    depth_target_mutual_latent, depth_target_mutual_pixel = depth_target_func(decoded_predict)

                if self.max_num_frame_for_decoding > 0:
                    # here we have to sample part of depth_latent to sent into AE decoder, for saving momery
                    depth_target_mutual_pixel = torch.index_select(depth_target_mutual_pixel, dim=0, index=sampled_idx)

                depth_target_dict['mutual_latent'] = depth_target_mutual_latent
                depth_target_dict['mutual_pixel'] = depth_target_mutual_pixel

            auxiliary_predict['depth'] = depth_predict_dict
            auxiliary_target['depth'] = depth_target_dict
        ############################ Perception branches ###########################

        return self.get_loss(predict, input, w, auxiliary_predict, auxiliary_target, **kwargs)

    def get_loss(
        self, 
        predict, 
        target, 
        w, 
        auxiliary_predict_dict: dict = None, 
        auxiliary_target_dict: dict = None,
        **kwargs,
    ):
        if self.loss_type == "l2":
            if self.use_additional_loss:
                predict_seq = rearrange(predict, "(b t) ... -> b t ...", t=self.num_frames)
                target_seq = rearrange(target, "(b t) ... -> b t ...", t=self.num_frames)
                bs = target.shape[0] // self.num_frames
                aux_loss = ((target_seq[:, 1:] - target_seq[:, :-1]) - (predict_seq[:, 1:] - predict_seq[:, :-1])) ** 2
                tmp_h, tmp_w = aux_loss.shape[-2], aux_loss.shape[-1]
                aux_loss = rearrange(aux_loss, "b t c h w -> b (t h w) c", c=4)
                aux_w = F.normalize(aux_loss, p=2)
                aux_w = rearrange(aux_w, "b (t h w) c -> b t c h w", t=self.num_frames - 1, h=tmp_h, w=tmp_w)
                aux_w = 1 + torch.cat((torch.zeros(bs, 1, *aux_w.shape[2:]).to(aux_w), aux_w), dim=1)
                aux_w = rearrange(aux_w, "b t ... -> (b t) ...").reshape(target.shape[0], -1)
                predict_hf = fourier_filter(predict, scale=0.)
                target_hf = fourier_filter(target, scale=0.)
                hf_loss = torch.mean((w * (predict_hf - target_hf) ** 2).reshape(target.shape[0], -1), 1).mean()

                loss = torch.mean(
                    (w * (predict - target) ** 2).reshape(target.shape[0], -1) * aux_w.detach(), 1
                ).mean() + self.additional_loss_weight * hf_loss
            else:
                return torch.mean(
                    (w * (predict - target) ** 2).reshape(target.shape[0], -1), 1
                )
        elif self.loss_type == "l1":
            if self.use_additional_loss:
                predict_seq = rearrange(predict, "(b t) ... -> b t ...", t=self.num_frames)
                target_seq = rearrange(target, "(b t) ... -> b t ...", t=self.num_frames)
                bs = target.shape[0] // self.num_frames
                aux_loss = ((target_seq[:, 1:] - target_seq[:, :-1]) - (predict_seq[:, 1:] - predict_seq[:, :-1])).abs()
                tmp_h, tmp_w = aux_loss.shape[-2], aux_loss.shape[-1]
                aux_loss = rearrange(aux_loss, "b t c h w -> b (t h w) c", c=4)
                aux_w = F.normalize(aux_loss, p=1)
                aux_w = rearrange(aux_w, "b (t h w) c -> b t c h w", t=self.num_frames - 1, h=tmp_h, w=tmp_w)
                aux_w = 1 + torch.cat((torch.zeros(bs, 1, *aux_w.shape[2:]).to(aux_w), aux_w), dim=1)
                aux_w = rearrange(aux_w, "b t ... -> (b t) ...").reshape(target.shape[0], -1)
                predict_hf = fourier_filter(predict, scale=0.)
                target_hf = fourier_filter(target, scale=0.)
                hf_loss = torch.mean((w * (predict_hf - target_hf).abs()).reshape(target.shape[0], -1), 1).mean()
                loss = torch.mean(
                    (w * (predict - target).abs()).reshape(target.shape[0], -1) * aux_w.detach(), 1
                ).mean() + self.additional_loss_weight * hf_loss
            else:
                return torch.mean(
                    (w * (predict - target).abs()).reshape(target.shape[0], -1), 1
                )
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")

        ######################### perception branches ############################
        if auxiliary_predict_dict:
            loss_dict = dict(loss_wo_perception=loss.mean())
            assert auxiliary_target_dict
            for key in auxiliary_predict_dict.keys():
                assert key in auxiliary_target_dict.keys()
                pred = auxiliary_predict_dict[key]
                tgt = auxiliary_target_dict[key]

                if key == 'depth':  # depth branch
                    depth_loss_pixel, depth_loss_latent = 0, 0
                    depth_mutual_loss_pixel, depth_mutual_loss_latent = 0, 0
                    if 'pixel' in self.perception_loss_mode:
                        pred_pixel = pred['pixel']
                        tgt_pixel = tgt['pixel']
                        valid_mask = torch.ones_like(pred_pixel)
                        depth_loss_pixel = self.depth_ssi_loss(pred_pixel, tgt_pixel, valid_mask)
                        if self.mutual_perception_loss:
                            depth_mutual_loss_pixel = self.depth_ssi_loss(pred_pixel, tgt['mutual_pixel'], valid_mask)
                    if 'latent' in self.perception_loss_mode:
                        pred_latent = pred['latent']
                        tgt_latent = tgt['latent']
                        depth_loss_latent = F.mse_loss(pred_latent, tgt_latent)
                        if self.mutual_perception_loss:
                            depth_mutual_loss_latent = F.mse_loss(pred_latent, tgt['mutual_latent'])

                    if isinstance(self.depth_loss_weight, dict):
                        pixel_weight = self.depth_loss_weight.get('pixel', 1.0)
                        latent_weight = self.depth_loss_weight.get('latent', 1.0)
                        pixel_mutual_weight = self.depth_loss_weight.get('pixel_mutual', 1.0)
                        latent_mutual_weight = self.depth_loss_weight.get('latent_mutual', 1.0)
                    else:
                        pixel_weight, latent_weight = self.depth_loss_weight, self.depth_loss_weight

                    depth_loss_pixel_weighted = pixel_weight * depth_loss_pixel
                    depth_loss_latent_weighted = latent_weight * depth_loss_latent
                    depth_loss = depth_loss_pixel_weighted + depth_loss_latent_weighted

                    if self.mutual_perception_loss:
                        depth_mutual_loss_pixel_weighted = pixel_mutual_weight * depth_mutual_loss_pixel
                        depth_mutual_loss_latent_weighted = latent_mutual_weight * depth_mutual_loss_latent

                        loss_dict['depth_mutual_loss_pixel_weighted'] = depth_mutual_loss_pixel_weighted
                        loss_dict['depth_mutual_loss_latent_weighted'] = depth_mutual_loss_latent_weighted

                        depth_loss += depth_mutual_loss_pixel_weighted + depth_mutual_loss_latent_weighted

                    loss_dict['depth_loss_pixel_weighted'] = depth_loss_pixel_weighted
                    loss_dict['depth_loss_latent_weighted'] = depth_loss_latent_weighted

                    if self.noise_level_perception_loss_weight:
                        depth_loss *= w.mean()

                    if self.use_additional_loss_for_depth:  # dynamic enhancement & structural preservation loss (Vista)
                        for mode in ['latent', 'pixel']:
                            if mode in self.perception_loss_mode:
                                pred_mode = pred[mode]
                                tgt_mode = tgt[mode]
                                # structural loss
                                depth_hf_loss = structure_preserving_loss(pred_mode, tgt_mode)
                                if self.noise_level_perception_loss_weight:
                                    depth_hf_loss *= w.mean()
                                loss_dict[f'depth_hf_loss_{mode}'] = depth_hf_loss
                                depth_loss += self.additional_loss_weight_depth * depth_hf_loss

                                # dynamic enhancement loss (Note: this loss need consecutive inputs)
                                if not (mode == 'pixel' and self.max_num_frame_for_decoding > 0):
                                    depth_dynamic_loss = dynamic_enhancement_loss(pred_mode, tgt_mode, self.num_frames)
                                    if self.noise_level_perception_loss_weight:
                                        depth_dynamic_loss *= w.mean()
                                    loss_dict[f'depth_dynamic_loss_{mode}'] = depth_dynamic_loss
                                    depth_loss += depth_dynamic_loss
                    loss_dict['depth_loss'] = depth_loss

                    if self.perception_loss_weight_schedulers is not None and 'depth' in self.perception_loss_weight_schedulers:
                        depth_loss_scheduler = self.perception_loss_weight_schedulers['depth']
                        global_step = kwargs.get('global_step', None)
                        assert global_step is not None, f'global_step is required for {type(depth_loss_scheduler)}'
                        total_depth_loss_weight = depth_loss_scheduler(global_step)
                        depth_loss *= total_depth_loss_weight
                        loss_dict['depth_loss_weighted'] = depth_loss

                    loss += depth_loss

            loss_dict['loss'] = loss
            return loss_dict
        ######################### perception branches ############################
        else:
            return loss



