from detectron2.config import CfgNode as CN
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
from detectron2.modeling.backbone.swin import SwinTransformer


def add_swin_config(cfg):
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.2
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_INDICES = [0, 1, 2, 3]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False


@BACKBONE_REGISTRY.register()
def build_swin_fpn_backbone(cfg, input_shape: ShapeSpec):
    bottom_up = SwinTransformer(
        pretrain_img_size=cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE,
        patch_size=cfg.MODEL.SWIN.PATCH_SIZE,
        in_chans=input_shape.channels,
        embed_dim=cfg.MODEL.SWIN.EMBED_DIM,
        depths=cfg.MODEL.SWIN.DEPTHS,
        num_heads=cfg.MODEL.SWIN.NUM_HEADS,
        window_size=cfg.MODEL.SWIN.WINDOW_SIZE,
        mlp_ratio=cfg.MODEL.SWIN.MLP_RATIO,
        qkv_bias=cfg.MODEL.SWIN.QKV_BIAS,
        drop_rate=cfg.MODEL.SWIN.DROP_RATE,
        attn_drop_rate=cfg.MODEL.SWIN.ATTN_DROP_RATE,
        drop_path_rate=cfg.MODEL.SWIN.DROP_PATH_RATE,
        ape=cfg.MODEL.SWIN.APE,
        patch_norm=cfg.MODEL.SWIN.PATCH_NORM,
        out_indices=cfg.MODEL.SWIN.OUT_INDICES,
        frozen_stages=cfg.MODEL.BACKBONE.FREEZE_AT,
        use_checkpoint=cfg.MODEL.SWIN.USE_CHECKPOINT,
    )
    return FPN(
        bottom_up=bottom_up,
        in_features=cfg.MODEL.FPN.IN_FEATURES,
        out_channels=cfg.MODEL.FPN.OUT_CHANNELS,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
