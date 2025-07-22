from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .depth_anything_v2.dpt import DPTHead, DepthAnythingV2, _make_fusion_block
from .depth_anything_v2.util.blocks import _make_scratch
from .depth_utils import get_2d_sincos_pos_embed, DepthTargetMappingBase

from unifuture.util import instantiate_from_config
from unifuture.modules.diffusionmodules.denoiser import Denoiser
from unifuture.modules.diffusionmodules.video_model import VideoUNet, timestep_embedding, repeat_as_img_seq
from unifuture.modules.diffusionmodules.util import normalization, zero_module
from unifuture.modules.diffusionmodules.openaimodel import Upsample


############################### for Perception branches ################################
class SequentialVideoUNetWithDepthBranch(VideoUNet):
    SUPPORT_PERCEPTION_MODE = ['align_latent', 'align_latent_multiscale']

    def __init__(
            self, 
            in_channels: int, 
            model_channels: int, 
            out_channels: int, 
            num_res_blocks: int, 
            attention_resolutions: int, 
            dropout: float = 0, 
            channel_mult: List[int] = (1, 2, 4, 8), 
            conv_resample: bool = True, 
            dims: int = 2, 
            num_classes: int = None, 
            use_checkpoint: bool = False, 
            num_heads: int = -1, 
            num_head_channels: int = -1, 
            num_heads_upsample: int = -1, 
            use_scale_shift_norm: bool = False, 
            resblock_updown: bool = False, 
            transformer_depth: List[int] = 1, 
            transformer_depth_middle: int = None, 
            context_dim: int = None, 
            time_downup: bool = False, 
            time_context_dim: int = None, 
            extra_ff_mix_layer: bool = False, 
            use_spatial_context: bool = False, 
            merge_strategy: str = "learned_with_images", 
            merge_factor: float = 0.5, 
            spatial_transformer_attn_type: str = "softmax", 
            video_kernel_size: List[int] = 3, 
            use_linear_in_transformer: bool = False, 
            adm_in_channels: int = None, 
            disable_temporal_crossattention: bool = False, 
            max_ddpm_temb_period: int = 10000, 
            add_lora: bool = False, 
            action_control: bool = False,
            # for depth branch
            perception_detach_mode = False,
            perception_detach_iter_range = (0, 1000000000),
            perception_mode = 'align_latent',
            perception_feedback = False,
            depth_branch_config = None,
    ):
        super().__init__(
            in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, 
            dropout, channel_mult, conv_resample, dims, num_classes, use_checkpoint, num_heads, 
            num_head_channels, num_heads_upsample, use_scale_shift_norm, resblock_updown, 
            transformer_depth, transformer_depth_middle, context_dim, time_downup, time_context_dim, 
            extra_ff_mix_layer, use_spatial_context, merge_strategy, merge_factor, 
            spatial_transformer_attn_type, video_kernel_size, use_linear_in_transformer, 
            adm_in_channels, disable_temporal_crossattention, max_ddpm_temb_period, add_lora, action_control
        )

        ######################## Perception branches ########################
        self.perception_detach_mode = perception_detach_mode
        self.perception_detach_ier_range = perception_detach_iter_range
        self.perception_mode = perception_mode
        self.perception_feedback = perception_feedback

        assert self.perception_mode in self.SUPPORT_PERCEPTION_MODE, f'only support perception_mode in: {self.SUPPORT_PERCEPTION_MODE}'
        if self.perception_detach_mode:
            print('*' * 20 + ' Using DETACHED perception branch mode!!!' + '*' * 20)
            print('*' * 20 + f' Will use perception detach mode between {self.perception_detach_ier_range} iter ' + '*' * 20)
        if self.perception_feedback:
            print('*' * 20 +  ' Using Perception Feedback Mode ' + '*' * 20)
            self.feedback_layer = zero_module(nn.Conv2d(4, 4, kernel_size=1, bias=False))   # zero init conv

        # 1. depth branch
        self.depth_branch = instantiate_from_config(depth_branch_config)
        ######################## Perception branches ########################

    def forward(
            self, 
            x: torch.Tensor, 
            timesteps: torch.Tensor, 
            context: torch.Tensor = None, 
            y: torch.Tensor = None, 
            time_context: torch.Tensor = None, 
            cond_mask: torch.Tensor = None, 
            num_frames: int = None,
            num_frames_override_for_perception: int = None,
    ):
        unet_predict = super().forward(x, timesteps, context, y, time_context, cond_mask, num_frames)

        # depth branch
        if self.perception_detach_mode:
            perception_input = unet_predict.clone().detach()
        else:
            perception_input = unet_predict

        if self.perception_mode == 'align_latent':
            depth_predict = self.depth_branch(perception_input)
        elif self.perception_mode == 'align_latent_multiscale':
            enc_hs, mid_hs, dec_hs = self.get_saved_intermediate(detach_mode=self.perception_detach_mode)
            depth_predict = self.depth_branch(mid_hs, enc_hs, dec_hs, num_frames_override=num_frames_override_for_perception)

        if self.perception_feedback:
            feedback = self.feedback_layer(depth_predict)
            unet_predict += feedback

        return unet_predict, depth_predict


class ParallelVideoUNetWithDepthBranch(VideoUNet):
    SUPPORT_PERCEPTION_MODE= ['align_latent_multiscale']

    def __init__(
            self, 
            in_channels: int, 
            model_channels: int, 
            out_channels: int, 
            num_res_blocks: int, 
            attention_resolutions: int, 
            dropout: float = 0, 
            channel_mult: List[int] = (1, 2, 4, 8),
            conv_resample: bool = True, 
            dims: int = 2, 
            num_classes: int = None, 
            use_checkpoint: bool = False, 
            num_heads: int = -1, 
            num_head_channels: int = -1, 
            num_heads_upsample: int = -1, 
            use_scale_shift_norm: bool = False, 
            resblock_updown: bool = False, 
            transformer_depth: List[int] = 1, 
            transformer_depth_middle: int = None, 
            context_dim: int = None, 
            time_downup: bool = False, 
            time_context_dim: int = None, 
            extra_ff_mix_layer: bool = False, 
            use_spatial_context: bool = False, 
            merge_strategy: str = "learned_with_images", 
            merge_factor: float = 0.5, 
            spatial_transformer_attn_type: str = "softmax", 
            video_kernel_size: int = 3, 
            use_linear_in_transformer: bool = False, 
            adm_in_channels: int = None, 
            disable_temporal_crossattention: bool = False, 
            max_ddpm_temb_period: int = 10000, 
            add_lora: bool = False, 
            action_control: bool = False,
            # for depth branch
            perception_detach_mode = False,
            perception_detach_iter_range = (0, 1000000000),
            perception_mode = 'align_latent',
            perception_feedback = False,
            depth_branch_config = None,
    ):
        super().__init__(
            in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, 
            dropout, channel_mult, conv_resample, dims, num_classes, use_checkpoint, num_heads, 
            num_head_channels, num_heads_upsample, use_scale_shift_norm, resblock_updown, 
            transformer_depth, transformer_depth_middle, context_dim, time_downup, time_context_dim, 
            extra_ff_mix_layer, use_spatial_context, merge_strategy, merge_factor, 
            spatial_transformer_attn_type, video_kernel_size, use_linear_in_transformer, 
            adm_in_channels, disable_temporal_crossattention, max_ddpm_temb_period, add_lora, action_control
        )

        ######################## Perception branches ########################
        self.perception_detach_mode = perception_detach_mode
        self.perception_detach_ier_range = perception_detach_iter_range
        self.perception_mode = perception_mode
        self.perception_feedback = perception_feedback

        assert self.perception_mode in self.SUPPORT_PERCEPTION_MODE, f'only support perception_mode in: {self.SUPPORT_PERCEPTION_MODE}'
        if self.perception_detach_mode:
            print('*' * 20 + ' Using DETACHED perception branch mode!!!' + '*' * 20)
            print('*' * 20 + f' Will use perception detach mode between {self.perception_detach_ier_range} iter ' + '*' * 20)
        if self.perception_feedback:
            print('*' * 20 +  ' Using Inside Perception Feedback Mode ' + '*' * 20)
            feedback_channels = [model_channels * mult for mult in channel_mult[::-1]]
            self.feedback_layer = nn.ModuleList([
                zero_module(nn.Conv2d(ch, ch, kernel_size=1, bias=False)) for ch in feedback_channels   # zero init conv
            ]) 

        # 1. depth branch
        self.depth_branch = instantiate_from_config(depth_branch_config)
        assert isinstance(self.depth_branch, MultiScaleDepthAlignMlp)
        ######################## Perception branches ########################

    def forward(
            self, x: torch.Tensor, 
            timesteps: torch.Tensor, 
            context: torch.Tensor = None, 
            y: torch.Tensor = None, 
            time_context: torch.Tensor = None, 
            cond_mask: torch.Tensor = None, 
            num_frames: int = None,
            num_frames_override_for_perception: int = None,
    ):
        # 1. the same part as the VideoUNet
        assert (y is not None) == (
                self.num_classes is not None
        ), "Must specify y if and only if the model is class-conditional"
        hs = list()
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        if cond_mask is not None and cond_mask.any():
            cond_mask_ = cond_mask[..., None].float()
            emb = self.cond_time_stack_embed(t_emb) * cond_mask_ + self.time_embed(t_emb) * (1 - cond_mask_)
        else:
            emb = self.time_embed(t_emb)

        if num_frames > 1 and context.shape[0] != x.shape[0]:
            assert context.shape[0] == x.shape[0] // num_frames, f"{context.shape} {x.shape}"
            context = repeat_as_img_seq(context, num_frames)

        if self.num_classes is not None:
            if num_frames > 1 and y.shape[0] != x.shape[0]:
                assert y.shape[0] == x.shape[0] // num_frames, f"{y.shape} {x.shape}"
                y = repeat_as_img_seq(y, num_frames)
            emb = emb + self.label_emb(y)

        h = x
        input_inters = list()
        # NOTE: we do NOT add feedback for the enc module, since we need the dec feats in depth branch
        for module in self.input_blocks:
            h = module(
                h,
                emb,
                context=context,
                time_context=time_context,
                num_frames=num_frames
            )
            hs.append(h)
            input_inters.append(h)
        input_inters = input_inters[self.num_res_blocks::(self.num_res_blocks + 1)]

        h = self.middle_block(
            h,
            emb,
            context=context,
            time_context=time_context,
            num_frames=num_frames
        )
        
        # 2. prepare for depth branch
        depth_latent = h
        depth_layer_idxes = list(range(len(self.output_blocks)))[(self.num_res_blocks - 1)::(self.num_res_blocks + 1)]
        depth_layer_idxes[-1] = len(self.output_blocks) - 1  # the biggest scale is little different

        # 3. decoder part of VideoUNet
        for i, module in enumerate(self.output_blocks):
            h = torch.cat((h, hs.pop()), dim=1)
            h = module(
                h,
                emb,
                context=context,
                time_context=time_context,
                num_frames=num_frames
            )

            # 4. for depth branch and feedback
            if i in depth_layer_idxes:
                scale_idx = depth_layer_idxes.index(i)
                current_enc = input_inters[-(scale_idx + 1)]
                current_dec = h
                
                depth_latent = self.depth_branch._forward_single_stage(
                    stage_id=scale_idx,
                    latents=depth_latent,
                    enc_h=current_enc,
                    dec_h=current_dec,
                )

                # feedback
                if self.perception_feedback:
                    feedback = self.feedback_layer[scale_idx](depth_latent)
                    h += feedback
                
                if scale_idx < len(depth_layer_idxes) - 1:  # the final scale does not need to upsample
                    depth_latent = self.depth_branch._forward_upsample(scale_idx, depth_latent)

        h = h.type(x.dtype)
        h_out = self.out(h)

        depth_predict = self.depth_branch._forward_output_proj(depth_latent)

        return h_out, depth_predict

    def set_save_intermediate(self, flag=True):
        pass  # do NOT need to save intermediate for parallel model


class DepthBranch(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_checkpoint=False,
    ) -> None:
        super().__init__()
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(  # 4x
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),       # 8x
            nn.Conv2d(           # 16x
                in_channels=out_channels[2],
                out_channels=out_channels[2],
                kernel_size=3,
                stride=2,
                padding=1),
            nn.Conv2d(           # 32x
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=5,
                stride=4,
                padding=1),
        ])
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )
        
        self.use_checkpoint = use_checkpoint
        if self.use_checkpoint:
            print(f"{self.__class__.__name__} is using checkpointing")

    def forward(self, latents: torch.Tensor, output_h, output_w, *args, **kwargs):
        if not self.use_checkpoint:
            return self._forward(latents, output_h, output_w)
        else:
            # NOTE (IMPORTANT): must set use_reentrant=False to avoid no gradinat bug when input.requires_grad=False 
            # (typically after detached)
            return checkpoint(self._forward, latents, output_h, output_w, use_reentrant=False)

    def _forward(self, latents: torch.Tensor, output_h, output_w):
        '''the forward pass of depth branch

        Args:
            latents (torch.Tensor): B x C x H x W

        Returns:
            depth: torch.Tensor, B x 1 x H x W
        '''        """"""
        out = []
        for proj_layer, resize_layer in zip(self.projects, self.resize_layers):
            x = proj_layer(latents)
            x = resize_layer(x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (output_h, output_w), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)

        out = F.relu(out)
        
        return out


class DepthAlignMlp(nn.Module):
    def __init__(
        self, 
        in_channels,
        num_features,
        hidden_depth=2,
        hidden_kernel=1,
        use_norm=True,
        positional_encoding=None,
        *args, 
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        modules = list()
        modules.append(nn.Conv2d(in_channels, num_features, kernel_size=1))  # input proj
        
        if positional_encoding is not None:
            pe_module = instantiate_from_config(positional_encoding)
            modules.append(pe_module)

        # hidden layers
        if isinstance(hidden_kernel, int):
            kernel_sizes = [hidden_kernel for _ in range(hidden_depth)]
        else:
            kernel_sizes = tuple(kernel_sizes)
            assert len(kernel_sizes) == hidden_depth
        padding_sizes = [(kernel - 1) // 2 for kernel in kernel_sizes]   # to keep the same output
        for i in range(hidden_depth):
            modules.append(nn.Conv2d(num_features, num_features, kernel_size=kernel_sizes[i], padding=padding_sizes[i]))
            if use_norm:
                modules.append(normalization(num_features))
            modules.append(nn.SiLU())
        
        # output proj
        modules.append(nn.Conv2d(num_features, in_channels, kernel_size=1))

        self.mlp = nn.Sequential(*modules)

    def forward(self, latents, *args, **kwargs):
        """latents: (B * T) x C x H x W"""
        return self.mlp(latents)


class MultiScaleDepthAlignMlp(nn.Module):
    def __init__(
            self, 
            out_channels,
            num_features, 
            num_scale=2, 
            num_block_each_stage=2,
            hidden_kernel=1, 
            use_norm=True, 
            multiscale_positional_encoding=None, 
            channel_mult=(1, 2),
            use_checkpoint=False,
            *args, 
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.use_checkpoint = use_checkpoint
        self.num_scale = num_scale

        # hidden layers
        if isinstance(hidden_kernel, int):
            kernel_sizes = [hidden_kernel for _ in range(num_scale)]
        else:
            kernel_sizes = tuple(hidden_kernel)
            assert len(kernel_sizes) == num_scale
        padding_sizes = [(kernel - 1) // 2 for kernel in kernel_sizes]   # to keep the same output
        assert len(channel_mult) == num_scale
        
        # PE
        if multiscale_positional_encoding is not None:
            self.latents_pe = instantiate_from_config(multiscale_positional_encoding)
            self.enc_hs_pe = instantiate_from_config(multiscale_positional_encoding)
            self.dec_hs_pe = instantiate_from_config(multiscale_positional_encoding)
        else:
            self.latents_pe = None 
            self.enc_hs_pe = None            
            self.dec_hs_pe = None

        # main body
        self.mlp = nn.ModuleList()
        self.input_fusion_conv = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for stage_i in range(num_scale):
            mult = channel_mult[stage_i]
            ks = kernel_sizes[stage_i]
            pad = padding_sizes[stage_i]
            ch = num_features * mult
            cur_module = []

            # for input fusion
            self.input_fusion_conv.append(nn.Conv2d(ch * 2, ch, kernel_size=1))

            for blk_idx in range(num_block_each_stage + 1):
                in_ch = ch * 2 if blk_idx == 0 else ch
                if use_norm:
                    cur_module.append(normalization(in_ch))
                cur_module.append(nn.SiLU())
                cur_module.append(nn.Conv2d(in_ch, ch, kernel_size=ks, padding=pad))

            if stage_i != num_scale - 1:
                self.upsamples.append(Upsample(ch, use_conv=True, dims=2, out_channels=num_features * channel_mult[stage_i + 1]))

            cur_module = nn.Sequential(*cur_module)
            self.mlp.append(cur_module)
                
        # output proj
        output_proj = list()
        if use_norm:
            output_proj.append(normalization(ch))
        output_proj.append(nn.SiLU())
        output_proj.append(nn.Conv2d(ch, out_channels, kernel_size=1))
        self.output_proj = nn.Sequential(*output_proj)

    def forward(self, latents, enc_hs, dec_hs, guider=None, num_frames_override=None, *args, **kwargs):
        if self.use_checkpoint:
            return checkpoint(self._forward, latents, enc_hs, dec_hs, guider, num_frames_override, use_reentrant=False)
        else:
            return self._forward(latents, enc_hs, dec_hs, guider, num_frames_override, *args, **kwargs)

    def _forward(self, latents, enc_hs, dec_hs, guider=None, num_frames_override=None, *args, **kwargs):
        '''forward pass for multi-scale depth mlp

        Args:
            latents (torch.Tensor): the final output from denoiser
            enc_hs (List[torch.Tensor]): the middle multi-scale features of denoiser encoder
            dec_hs (List[torch.Tensor]): the middle multi-scale features of denoiser decoder
            guider (Guider): the guider instance that get final feats from unconditional and conditional features
            num_frames_override (Optional[int]): override the num_frames in guider (useful for training)

        Returns:
            depth_latents (torch.Tensor): predicted depth latent
        '''
        if guider is not None and hasattr(guider, 'num_frames') and num_frames_override is not None:
            ori_num_frames = guider.num_frames
            guider.num_frames = num_frames_override

        # 1. input proj
        if guider is not None:
            latents = guider(latents, sigma=None)

        assert len(enc_hs) == len(dec_hs) and len(enc_hs) == len(self.mlp), f'the num of levels must match'
        # 2. hidden layers
        for i in range(self.num_scale):
            enc_in = enc_hs[-(i + 1)].float()   # B x C x H x W
            dec_in = dec_hs[i].float()
            # NOTE: we need to merge [unconditional, conditional] into the final feature, so we use guider here
            if guider is not None:
                enc_in, dec_in = guider(enc_in, sigma=None), guider(dec_in, sigma=None)   

            latents = self._forward_single_stage(i, latents, enc_in, dec_in)
            if i < self.num_scale - 1:
                latents = self._forward_upsample(i, latents)

        # 3. output proj
        out = self._forward_output_proj(latents)

        if guider is not None and hasattr(guider, 'num_frames') and num_frames_override is not None:
            guider.num_frames = ori_num_frames

        return out

    def _forward_single_stage(self, stage_id, latents, enc_h, dec_h, *args, **kwargs):
        '''forward pass for singe scale
        '''
        enc_h = enc_h.float()
        dec_h = dec_h.float()
        if self.enc_hs_pe is not None and self.dec_hs_pe is not None:
            enc_in = self.enc_hs_pe(enc_h, level_id=stage_id)
            dec_in = self.dec_hs_pe(dec_h, level_id=stage_id)
        
        in_fusion_layer = self.input_fusion_conv[stage_id]
        fused_in = in_fusion_layer(torch.cat([enc_in, dec_in], dim=1))

        if self.latents_pe is not None:
            latents = self.latents_pe(latents, level_id=stage_id)
        latents = torch.cat([latents.float(), fused_in], dim=1)
        layer = self.mlp[stage_id]
        latents = layer(latents)

        return latents

    def _forward_upsample(self, stage_id, latents, *args, **kwargs):
        return self.upsamples[stage_id](latents)

    def _forward_output_proj(self, latents, *args, **kwargs):
        '''forward pass for output projection
        '''
        return self.output_proj(latents)

class DepthTargetEngine(nn.Module):
    def __init__(
        self, 
        engine_config,
        ckpt_path=None,
        depth_mapping_config=None,
        *args, 
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.depth_target_engine: nn.Module = instantiate_from_config(engine_config)
        self.depth_mapping: DepthTargetMappingBase = instantiate_from_config(depth_mapping_config) if depth_mapping_config is not None else None

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    @torch.no_grad()
    def forward(self, image, infer_size=None):
        '''generate the depth target using off-the-shelf depth estimator (engine)

        Args:
            image (torch.Tensor): the input image of shape B x 3 x H x W

        Returns:
            depth_target (torch.Tensor): B x 1 x H x W
        '''
        if infer_size is not None:
            assert isinstance(infer_size, int)
            raw_h, raw_w = image.shape[-2:]
            image = F.interpolate(image, (infer_size, infer_size), mode='bilinear', align_corners=True)

        depth_target = self.depth_target_engine(image)
        if len(depth_target.shape) == 3:
            depth_target = depth_target.unsqueeze(1)
        
        if infer_size is not None:
            depth_target = F.interpolate(depth_target, (raw_h, raw_w), mode='bilinear', align_corners=True)

        if self.depth_mapping is not None:
            depth_target, _, _ = self.depth_mapping.map(depth_target) 

        return depth_target

    def init_from_ckpt(self, ckpt_path):
        print('*' * 20 + f" Now loading weights of {self.__class__.__name__} " + '*' * 20)
        self.depth_target_engine.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=True)   
        print('*' * 20 + f" Loading weights finished " + '*' * 20)

