from unifuture.modules.attention import *
from unifuture.modules.diffusionmodules.util import AlphaBlender, linear, timestep_embedding


class TimeMixSequential(nn.Sequential):
    def forward(self, x, context=None, timesteps=None):
        for layer in self:
            x = layer(x, context, timesteps)
        return x


class VideoTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention  # ampere
    }

    def __init__(
            self,
            dim,
            n_heads,
            d_head,
            dropout=0.0,
            context_dim=None,
            gated_ff=True,
            use_checkpoint=False,
            timesteps=None,
            ff_in=False,
            inner_dim=None,
            attn_mode="softmax",
            disable_self_attn=False,
            disable_temporal_crossattention=False,
            switch_temporal_ca_to_sa=False,
            add_lora=False,
            action_control=False
    ):
        super().__init__()
        attn_cls = self.ATTENTION_MODES[attn_mode]

        self.ff_in = ff_in or inner_dim is not None
        if inner_dim is None:
            inner_dim = dim

        assert int(n_heads * d_head) == inner_dim

        self.is_res = inner_dim == dim

        if self.ff_in:
            self.norm_in = nn.LayerNorm(dim)
            self.ff_in = FeedForward(dim, dim_out=inner_dim, dropout=dropout, glu=gated_ff)

        self.timesteps = timesteps
        self.disable_self_attn = disable_self_attn
        if disable_self_attn:
            self.attn1 = attn_cls(
                query_dim=inner_dim,
                context_dim=context_dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                add_lora=add_lora
            )  # is a cross-attn
        else:
            self.attn1 = attn_cls(
                query_dim=inner_dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                causal=False,
                add_lora=add_lora
            )  # is a self-attn

        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout, glu=gated_ff)

        if not disable_temporal_crossattention:
            self.norm2 = nn.LayerNorm(inner_dim)
            if switch_temporal_ca_to_sa:
                self.attn2 = attn_cls(
                    query_dim=inner_dim,
                    heads=n_heads,
                    dim_head=d_head,
                    dropout=dropout,
                    causal=False,
                    add_lora=add_lora
                )  # is a self-attn
            else:
                self.attn2 = attn_cls(
                    query_dim=inner_dim,
                    context_dim=context_dim,
                    heads=n_heads,
                    dim_head=d_head,
                    dropout=dropout,
                    add_lora=add_lora,
                    action_control=action_control
                )  # is self-attn if context is None

        self.norm1 = nn.LayerNorm(inner_dim)
        self.norm3 = nn.LayerNorm(inner_dim)
        self.switch_temporal_ca_to_sa = switch_temporal_ca_to_sa

        self.use_checkpoint = use_checkpoint
        if self.use_checkpoint:
            print(f"{self.__class__.__name__} is using checkpointing")

    def forward(self, x: torch.Tensor, context: torch.Tensor = None, timesteps: int = None) -> torch.Tensor:
        if self.use_checkpoint:
            return checkpoint(self._forward, x, context, timesteps)
        else:
            return self._forward(x, context, timesteps=timesteps)

    def _forward(self, x, context=None, timesteps=None):
        assert self.timesteps or timesteps
        assert not (self.timesteps and timesteps) or self.timesteps == timesteps
        timesteps = self.timesteps or timesteps
        B, S, C = x.shape
        x = rearrange(x, "(b t) s c -> (b s) t c", t=timesteps)

        if self.ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))
            if self.is_res:
                x += x_skip

        if self.disable_self_attn:
            x = self.attn1(self.norm1(x), context=context, batchify_xformers=True) + x
        else:  # this way
            x = self.attn1(self.norm1(x), batchify_xformers=True) + x

        if hasattr(self, "attn2"):
            if self.switch_temporal_ca_to_sa:
                x = self.attn2(self.norm2(x), batchify_xformers=True) + x
            else:  # this way
                x = self.attn2(self.norm2(x), context=context, batchify_xformers=True) + x

        x_skip = x
        x = self.ff(self.norm3(x))
        if self.is_res:
            x += x_skip

        x = rearrange(x, "(b s) t c -> (b t) s c", s=S, b=B // timesteps, c=C, t=timesteps)
        return x

    def get_last_layer(self):
        return self.ff.net[-1].weight


class SpatialVideoTransformer(SpatialTransformer):
    def __init__(
            self,
            in_channels,
            n_heads,
            d_head,
            depth=1,
            dropout=0.0,
            use_linear=False,
            context_dim=None,
            use_spatial_context=False,
            timesteps=None,
            merge_strategy: str = "fixed",
            merge_factor: float = 0.5,
            time_context_dim=None,
            ff_in=False,
            use_checkpoint=False,
            time_depth=1,
            attn_mode="softmax",
            disable_self_attn=False,
            disable_temporal_crossattention=False,
            max_time_embed_period=10000,
            add_lora=False,
            action_control=False
    ):
        super().__init__(
            in_channels,
            n_heads,
            d_head,
            depth=depth,
            dropout=dropout,
            attn_type=attn_mode,
            use_checkpoint=use_checkpoint,
            context_dim=context_dim,
            use_linear=use_linear,
            disable_self_attn=disable_self_attn,
            add_lora=add_lora,
            action_control=action_control
        )
        self.time_depth = time_depth
        self.depth = depth
        self.max_time_embed_period = max_time_embed_period

        time_mix_d_head = d_head
        n_time_mix_heads = n_heads

        time_mix_inner_dim = int(time_mix_d_head * n_time_mix_heads)

        inner_dim = n_heads * d_head
        if use_spatial_context:
            time_context_dim = context_dim

        self.time_stack = nn.ModuleList(
            [
                VideoTransformerBlock(
                    inner_dim,
                    n_time_mix_heads,
                    time_mix_d_head,
                    dropout=dropout,
                    context_dim=time_context_dim,
                    timesteps=timesteps,
                    use_checkpoint=use_checkpoint,
                    ff_in=ff_in,
                    inner_dim=time_mix_inner_dim,
                    attn_mode=attn_mode,
                    disable_self_attn=disable_self_attn,
                    disable_temporal_crossattention=disable_temporal_crossattention,
                    add_lora=add_lora,
                    action_control=action_control
                )
                for _ in range(self.depth)
            ]
        )

        assert len(self.time_stack) == len(self.transformer_blocks)

        self.use_spatial_context = use_spatial_context
        self.in_channels = in_channels

        time_embed_dim = in_channels * 4
        self.time_pos_embed = nn.Sequential(
            linear(in_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, in_channels)
        )

        self.time_mixer = AlphaBlender(
            alpha=merge_factor,
            merge_strategy=merge_strategy,
            rearrange_pattern="b t -> (b t) 1 1"
        )

    def forward(
            self,
            x: torch.Tensor,
            context: Optional[torch.Tensor] = None,
            time_context: Optional[torch.Tensor] = None,
            timesteps: Optional[int] = None
    ) -> torch.Tensor:
        _, _, h, w = x.shape
        x_in = x
        spatial_context = None
        if exists(context):
            spatial_context = context

        if self.use_spatial_context:
            assert context.ndim == 3, f"Dims of spatial context should be 3 but are {context.ndim}"

            time_context = context
            time_context_first_timestep = time_context[::timesteps]
            time_context = repeat(time_context_first_timestep, "b ... -> (b n) ...", n=h * w)
        elif time_context is not None and not self.use_spatial_context:
            time_context = repeat(time_context, "b ... -> (b n) ...", n=h * w)
            if time_context.ndim == 2:
                time_context = rearrange(time_context, "b c -> b 1 c")

        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.use_linear:
            x = self.proj_in(x)

        num_frames = torch.arange(timesteps, device=x.device)
        num_frames = repeat(num_frames, "t -> (b t)", b=x.shape[0] // timesteps)
        t_emb = timestep_embedding(
            num_frames,
            self.in_channels,
            repeat_only=False,
            max_period=self.max_time_embed_period
        )
        emb = self.time_pos_embed(t_emb)
        emb = emb[:, None]

        for block, mix_block in zip(self.transformer_blocks, self.time_stack):
            x = block(x, context=spatial_context)

            x_mix = x
            x_mix = x_mix + emb

            x_mix = mix_block(x_mix, context=time_context, timesteps=timesteps)
            x = self.time_mixer(x_spatial=x, x_temporal=x_mix)

        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        if not self.use_linear:
            x = self.proj_out(x)
        out = x + x_in
        return out
