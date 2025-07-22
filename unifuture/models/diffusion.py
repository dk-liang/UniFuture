import math
import matplotlib
import cv2
import numpy as np
import os
import os.path as osp
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple, Union
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from omegaconf import ListConfig, OmegaConf
from pytorch_lightning import LightningModule
from safetensors.torch import load_file as load_safetensors
from torch.optim.lr_scheduler import LambdaLR

from unifuture.modules import UNCONDITIONAL_CONFIG
from unifuture.modules.autoencoding.temporal_ae import VideoDecoder
from unifuture.modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from unifuture.modules.diffusionmodules.util import zero_module
from unifuture.modules.ema import LitEma
from unifuture.modules.perception.depth_utils import get_depth_vis, split_tensor_list
from unifuture.util import default, disabled_train, get_obj_from_str, instantiate_from_config


class DiffusionEngine(LightningModule):
    SUPPORT_PERCEPTION_MODE = ['align_latent', 'align_latent_multiscale']

    def __init__(
            self,
            network_config,
            denoiser_config,
            first_stage_config,
            conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
            sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
            optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
            scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
            loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
            network_wrapper: Union[None, str] = None,
            ckpt_path: Union[None, str] = None,
            use_ema: bool = False,
            ema_decay_rate: float = 0.9999,
            scale_factor: float = 1.0,
            disable_first_stage_autocast=False,
            input_key: str = "img",
            log_keys: Union[List, None] = None,
            no_cond_log: bool = False,
            compile_model: bool = False,
            en_and_decode_n_samples_a_time: int = 14,
            num_frames: int = 25,
            slow_spatial_layers: bool = False,
            train_peft_adapters: bool = False,
            replace_cond_frames: bool = False,
            fixed_cond_frames: Union[List, None] = None,
            perception_mem_save_mode = False,
            perception_mode = 'align_latent',
            perception_feedback = False,
            train_perception_only = False,
            depth_target_engine_config: Union[None, Dict, ListConfig, OmegaConf] = None,
            depth_lr_coeff = 1.0,
    ):
        super().__init__()
        self.log_keys = log_keys
        self.input_key = input_key
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.AdamW"}
        )
        model = instantiate_from_config(network_config)
        self.model = get_obj_from_str(
            default(network_wrapper, OPENAIUNETWRAPPER)
        )(
            model, compile_model=compile_model
        )

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = (
            instantiate_from_config(sampler_config)
            if sampler_config is not None
            else None
        )
        self.conditioner = instantiate_from_config(
            default(conditioner_config, UNCONDITIONAL_CONFIG)
        )
        self.scheduler_config = scheduler_config
        self._init_first_stage(first_stage_config)

        self.loss_fn = (
            instantiate_from_config(loss_fn_config)
            if loss_fn_config is not None
            else None
        )

        ######################## Perception branches ########################
        self.perception_mem_save_mode = perception_mem_save_mode
        self.perception_mode = perception_mode
        self.perception_feedback = perception_feedback
        self.train_perception_only = train_perception_only

        assert self.perception_mode in self.SUPPORT_PERCEPTION_MODE, f'only support perception_mode in: {self.SUPPORT_PERCEPTION_MODE}'
        if self.perception_feedback:
            print('*' * 20 +  ' Using Outside Perception Feedback Mode ' + '*' * 20)
            self.feedback_layer = zero_module(nn.Conv2d(4, 4, kernel_size=1, bias=False))   # zero init conv

        self.depth_lr_coeff = depth_lr_coeff
        self.depth_target_engine = instantiate_from_config(depth_target_engine_config) if depth_target_engine_config else None
        if self.depth_target_engine:
            for p in self.depth_target_engine.parameters():
                p.requires_grad = False

        ######################## Perception branches ########################

        self.use_ema = use_ema
        self.ema_decay_rate = ema_decay_rate
        if use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay_rate)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}")

        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
        self.num_frames = num_frames
        self.slow_spatial_layers = slow_spatial_layers
        self.train_peft_adapters = train_peft_adapters
        self.replace_cond_frames = replace_cond_frames
        self.fixed_cond_frames = fixed_cond_frames

    def reinit_ema(self):
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=self.ema_decay_rate)
            print(f"Reinitializing EMAs of {len(list(self.model_ema.buffers()))}")

    def init_from_ckpt(self, path: str) -> None:
        if path.endswith("ckpt"):
            svd = torch.load(path, map_location="cpu")["state_dict"]
        elif path.endswith("bin"):  # for deepspeed merged checkpoints
            svd = torch.load(path, map_location="cpu")
            for k in list(svd.keys()):  # remove the prefix
                if "_forward_module" in k:
                    svd[k.replace("_forward_module.", "")] = svd[k]
                del svd[k]
        elif path.endswith("safetensors"):
            svd = load_safetensors(path)
        else:
            raise NotImplementedError

        missing, unexpected = self.load_state_dict(svd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected keys: {unexpected}")

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict
        # image tensors should be scaled to -1 ... 1 and in bchw format
        input_shape = batch[self.input_key].shape
        if len(input_shape) != 4:  # is an image sequence
            assert input_shape[1] == self.num_frames
            batch[self.input_key] = rearrange(batch[self.input_key], "b t c h w -> (b t) c h w")
        return batch[self.input_key]

    def decode_first_stage(self, z, overlap=3, depth_z=None, no_grad=True, use_checkpoint=False):
        # we need this function to support pixel loss in align_latent mode while preserving compability
        if no_grad:
            with torch.no_grad():
                return self._decode_first_stage(z, overlap, depth_z)
        else:
            if use_checkpoint:
                return checkpoint(self._decode_first_stage, z, overlap, depth_z)
            else:
                return self._decode_first_stage(z, overlap, depth_z)

    def _decode_first_stage(self, z, overlap=3, depth_z=None):
        if depth_z is not None and self.perception_feedback:
            feedback = self.feedback_layer(depth_z)
            z += feedback

        z = z / self.scale_factor
        if depth_z is not None:  # for depth branch
            depth_z = depth_z / self.scale_factor

        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])
        all_out = list()
        all_depth_out = list()
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            if overlap < n_samples:
                previous_z = z[:overlap]
                if depth_z is not None:
                    previous_depth_z = depth_z[:overlap]
                    depth_z_slices = depth_z[overlap:].split(n_samples - overlap, dim=0)

                for chunk_id, current_z in enumerate(z[overlap:].split(n_samples - overlap, dim=0)):
                    if isinstance(self.first_stage_model.decoder, VideoDecoder):
                        kwargs = {"timesteps": current_z.shape[0] + overlap}
                    else:
                        kwargs = dict()
                    context_z = torch.cat((previous_z, current_z), dim=0)
                    previous_z = current_z[-overlap:]
                    out = self.first_stage_model.decode(context_z, **kwargs)

                    if depth_z is not None:
                        current_depth_z = depth_z_slices[chunk_id]
                        context_depth_z = torch.cat((previous_depth_z, current_depth_z), dim=0)
                        previous_depth_z = current_depth_z[-overlap:]
                        depth_out = self.first_stage_model.decode(context_depth_z, **kwargs)
                        depth_out = torch.mean(depth_out, dim=1, keepdim=True)  # B x 1 x H x W

                    if not all_out:
                        all_out.append(out)
                        if depth_z is not None:
                            all_depth_out.append(depth_out)
                    else:
                        all_out[-1][-overlap:] = (all_out[-1][-overlap:] + out[:overlap]) / 2
                        all_out.append(out[overlap:])
                        if depth_z is not None:
                            all_depth_out[-1][-overlap:] = (all_depth_out[-1][-overlap:] + depth_out[:overlap]) / 2
                            all_depth_out.append(depth_out[overlap:])
            else:
                if depth_z is not None:
                    depth_z_slices = depth_z.split(n_samples, dim=0)

                for i, current_z in enumerate(z.split(n_samples, dim=0)):
                    if isinstance(self.first_stage_model.decoder, VideoDecoder):
                        kwargs = {"timesteps": current_z.shape[0]}
                    else:
                        kwargs = dict()

                    ########################## perception branches ######################

                    if depth_z is not None:
                        current_depth_z = depth_z_slices[i]
                        depth_out = self.first_stage_model.decode(current_depth_z, **kwargs)
                        all_depth_out.append(depth_out)

                    ########################## perception branches ######################

                    out = self.first_stage_model.decode(current_z, **kwargs)
                    all_out.append(out)

        out = torch.cat(all_out, dim=0)

        if depth_z is not None:
            depth = torch.cat(all_depth_out) if len(all_depth_out) else None
            depth = torch.mean(depth, dim=1, keepdim=True)  # B x 1 x H x W
            return out, depth
        else:
            return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = list()
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.encode(
                    x[n * n_samples: (n + 1) * n_samples]
                )
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = z * self.scale_factor
        return z

    def forward(self, x, batch, depth_target=None, depth_target_pixel=None):
        loss = self.loss_fn(
            self.model, 
            self.denoiser, 
            self.conditioner, 
            x, 
            batch, 
            depth_target,
            perception_mode=self.perception_mode,
            perception_feedback = self.perception_feedback,
            perception_feedback_layer = self.feedback_layer if hasattr(self, 'feedback_layer') else None,
            dec_func=partial(self.decode_first_stage, no_grad=False, use_checkpoint=True),
            depth_target_pixel=depth_target_pixel,
            depth_target_func=self.get_depth_target,
            global_step=self.global_step,
        )  # go to StandardDiffusionLoss
        if isinstance(loss, dict):
            loss_dict = loss
            loss_mean = loss_dict['loss'].mean()
        else:
            loss_mean = loss.mean()
            loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    @torch.no_grad()
    def get_depth_target(self, img):
        if self.depth_target_engine is not None:
            if not self.perception_mem_save_mode:
                depth_target_pixel = self.depth_target_engine(img, infer_size=518) 
            else:   # save memory
                all_target_list = list()
                for single_x in torch.split(img, 1, dim=0):
                    all_target_list.append(self.depth_target_engine(single_x, infer_size=518))
                depth_target_pixel = torch.cat(all_target_list, dim=0)

            if self.perception_mode.startswith('align_latent'):
                depth_target_remapped = depth_target_pixel.repeat(1, 3, 1, 1)  # to B x 3 x H x W
                depth_target_latent = self.encode_first_stage(depth_target_remapped)
        else:
            depth_target_pixel = None
            depth_target_latent = None

        return depth_target_latent, depth_target_pixel

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch)

        ################ perception branches ##################
        if self.training:
            depth_target_latent, depth_target_pixel = self.get_depth_target(x)
        else:
            depth_target_latent, depth_target_pixel = None, None

        if self.perception_mode.startswith('align_latent'):
            depth_target = depth_target_latent
        else:
            depth_target = depth_target_pixel
        ################ perception branches ##################

        x = self.encode_first_stage(x)
        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch, depth_target, depth_target_pixel)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)

    def on_train_start(self, *args, **kwargs):
        if self.sampler is None or self.loss_fn is None:
            raise ValueError("Sampler and loss function need to be set for training")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def configure_optimizers(self):
        lr = self.learning_rate
        if self.train_perception_only:
            param_dicts = [
                {
                    "params": [p for n, p in self.model.named_parameters() if "depth" in n or 'feedback' in n],
                    "lr": lr * self.depth_lr_coeff
                }
            ]
            if self.perception_feedback:
                param_dicts.append(
                    {
                        "params": list(self.feedback_layer.parameters()),
                    }
                )
            print('*' * 20 + f' ONLY TRAIN PERCEPTION BRANCHES, Lr of depth branch: {lr * self.depth_lr_coeff} ' + '*' * 20)
        else:
            if self.slow_spatial_layers:
                param_dicts = [
                    {
                        "params": [p for n, p in self.model.named_parameters() if "time_stack" in n]
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if "time_stack" not in n],
                        "lr": lr * 0.1
                    }
                ]
            elif self.train_peft_adapters:
                param_dicts = [
                    {
                        "params": [p for n, p in self.model.named_parameters() if "adapter" in n]
                    }
                ]
            else:
                param_dicts = [
                    {
                        "params": list(self.model.parameters())
                    }
                ]
            for embedder in self.conditioner.embedders:
                if embedder.is_trainable:
                    param_dicts.append(
                        {
                            "params": list(embedder.parameters())
                        }
                    )
            ############################ Perception branches ################################

            if self.perception_feedback:
                param_dicts.append(
                    {
                        "params": list(self.feedback_layer.parameters())
                    }
                )

            ############################ Perception branches ################################
    
        opt = self.instantiate_optimizer_from_config(param_dicts, lr, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1
                }
            ]
            return [opt], scheduler
        else:
            return opt

    @torch.no_grad()
    def sample(
            self,
            cond: Dict,
            cond_frame=None,
            uc: Union[Dict, None] = None,
            N: int = 25,
            shape: Union[None, Tuple, List] = None,
            **kwargs
    ):
        randn = torch.randn(N, *shape).to(self.device)
        cond_mask = torch.zeros(N).to(self.device)
        if self.replace_cond_frames:
            assert self.fixed_cond_frames
            cond_indices = self.fixed_cond_frames
            cond_mask = rearrange(cond_mask, "(b t) -> b t", t=self.num_frames)
            cond_mask[:, cond_indices] = 1
            cond_mask = rearrange(cond_mask, "b t -> (b t)")

        self.model.diffusion_model.set_save_intermediate(flag=True)
        denoiser = lambda input, sigma, c, cond_mask: self.denoiser(self.model, input, sigma, c, cond_mask, **kwargs)
        samples, depth = self.sampler(  # go to EulerEDMSampler
            denoiser, randn, cond, uc=uc, cond_frame=cond_frame, cond_mask=cond_mask
        )
        self.model.diffusion_model.set_save_intermediate(flag=False)
        return samples, depth

    @torch.no_grad()
    def log_images(
            self,
            batch: Dict,
            N: int = 25,
            sample: bool = True,
            ucg_keys: List[str] = None,
            **kwargs
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders if e.ucg_rate > 0.0]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys, "
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x = self.get_input(batch)

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else list()
        )

        sampling_kwargs = dict()

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]

        ################ perception branches ##################
        with torch.no_grad():
            if self.depth_target_engine is None:
                depth_target = None
            else:
                depth_target_z, depth_target = self.get_depth_target(x)
            if depth_target is not None:
                depth_target_vis = get_depth_vis(img=None, depth=depth_target, to_numpy=False)
                depth_target_vis = torch.stack(depth_target_vis)
                log["depth_target"] = depth_target_vis
                if depth_target_z is not None:
                    depth_rec = self.decode_first_stage(depth_target_z)
                    depth_rec = torch.mean(depth_rec, dim=1, keepdim=True)  # B x 1 x H x W
                    depth_rec = get_depth_vis(img=None, depth=depth_rec, to_numpy=False)
                    depth_rec = torch.stack(depth_rec)
                    log["depth_target_rec_from_latent"] = depth_rec
        ################ perception branches ##################

        z = self.encode_first_stage(x)
        x_reconstruct = self.decode_first_stage(z)

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))
                if c[k].shape[0] < N:
                    c[k] = c[k][[0]]
                if uc[k].shape[0] < N:
                    uc[k] = uc[k][[0]]

        if sample:
            with self.ema_scope("Plotting"):
                samples, depth_z = self.sample(
                    c, cond_frame=z, shape=z.shape[1:], uc=uc, N=N, **sampling_kwargs
                )

            samples, depth = self.decode_first_stage(samples, depth_z=depth_z)
            log["samples"] = log["samples_mp4"] = samples

            if depth is not None:
                depth_vis = get_depth_vis(img=None, depth=depth, to_numpy=False)
                depth_vis = torch.stack(depth_vis)
                log["depth_decode_func"] = depth_vis

        log["inputs"] = log["inputs_mp4"] = x
        log["targets"] = log["targets_mp4"] = x_reconstruct
        return log
