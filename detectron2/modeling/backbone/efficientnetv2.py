import torch
import torch.nn as nn
from detectron2.modeling.backbone import Backbone, FPN
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
import timm

class EfficientNetV2Backbone(Backbone):
    """
    Implementasi EfficientNetV2 sebagai backbone untuk Detectron2.
    """
    def __init__(self, timm_model, out_features):
        super().__init__()
        self.model = timm_model
        
        self._out_features = out_features
        
        feature_info = self.model.feature_info
        
        self._out_feature_channels = {
            "res2": feature_info[0]['num_chs'], # Stride 4
            "res3": feature_info[1]['num_chs'], # Stride 8
            "res4": feature_info[2]['num_chs'], # Stride 16
            "res5": feature_info[3]['num_chs'], # Stride 32
        }
        
        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }

    def forward(self, x):
        features = self.model(x)
        outputs = {}
        for i, name in enumerate(self._out_features):
            outputs[name] = features[i]
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_efficientnetv2_s_backbone(cfg, input_shape):
    """
    Membangun EfficientNetV2-S backbone.
    """
    # Initialize timm model with ImageNet-1K pre-trained weights
    # out_indices=(1, 2, 3, 4) corresponds to strides (4, 8, 16, 32)
    timm_model = timm.create_model(
        'tf_efficientnetv2_s.in21k_ft_in1k', # Menggunakan versi pretrained ImageNet-1K yang akurat
        pretrained=True, 
        features_only=True, 
        out_indices=(1, 2, 3, 4) 
    )
    
    out_features = ["res2", "res3", "res4", "res5"]
    return EfficientNetV2Backbone(timm_model, out_features)

@BACKBONE_REGISTRY.register()
def build_efficientnetv2_s_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Membangun EfficientNetV2-S dengan FPN.
    """
    bottom_up = build_efficientnetv2_s_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
