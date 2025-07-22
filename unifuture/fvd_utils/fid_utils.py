from typing import Union, Tuple, Any
import torch
from torch import Tensor
from torch.nn import Module
from torchmetrics.image.fid import FrechetInceptionDistance


class FlexibleFrechetInceptionDistance(FrechetInceptionDistance):
    def __init__(
        self,
        feature: Union[int, Module] = 2048,
        reset_real_features: bool = True,
        normalize: bool = False,
        input_img_size: Tuple[int, int, int] = (3, 299, 299),
        **kwargs: Any,
    ) -> None:
        super().__init__(feature, reset_real_features, normalize, input_img_size, **kwargs)

    def extract_feats(self, imgs: Tensor) -> Tensor:
        imgs = (imgs * 255).byte() if self.normalize and (not self.used_custom_model) else imgs
        features = self.inception(imgs)
        return features

    def update_with_features(self, features, real: bool) -> None:
        self.orig_dtype = features.dtype
        features = features.double()

        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += features.shape[0]
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += features.shape[0]

    def update(self, imgs: Tensor, real: bool, skip_feat_extraction=False) -> None:
        if not skip_feat_extraction:
            features = self.extract_feats(imgs)
        else:
            features = imgs

        self.update_with_features(features, real)