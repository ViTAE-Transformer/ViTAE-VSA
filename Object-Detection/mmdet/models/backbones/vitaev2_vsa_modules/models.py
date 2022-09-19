"""
T2T-ViT
"""
from math import gamma
import torch
import torch.nn as nn

from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
import numpy as np
from .base_model import ViTAE_DeformWindow_NoShift_basic

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'ViTAE_stages3_7': _cfg(),
}


@register_model
def ViTAE_DeformWindow_NoShift_12_basic_stages4_14(pretrained=False, **kwargs): # adopt performer for tokens to token
    # if pretrained:
        # kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = ViTAE_DeformWindow_NoShift_basic(RC_tokens_type=['VSA', 'VSA', 'transformer', 'transformer'], NC_tokens_type=['VSA', 'VSA', 'transformer', 'transformer'], stages=4, embed_dims=[64, 64, 128, 256], token_dims=[64, 128, 256, 512], downsample_ratios=[4, 2, 2, 2],
                            NC_depth=[2, 2, 8, 2], NC_heads=[1, 2, 4, 8], RC_heads=[1, 1, 2, 4], mlp_ratio=4., NC_group=[1, 32, 64, 128], RC_group=[1, 16, 32, 64], window_size=7, **kwargs)
    model.default_cfg = default_cfgs['ViTAE_stages3_7']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def ViTAE_DeformWindow_NoShift_1234_basic_stages4_14(pretrained=False, **kwargs): # adopt performer for tokens to token
    # if pretrained:
        # kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = ViTAE_DeformWindow_NoShift_basic(RC_tokens_type=['VSA', 'VSA', 'VSA', 'VSA'], NC_tokens_type=['VSA', 'VSA', 'VSA', 'VSA'], stages=4, embed_dims=[64, 64, 128, 256], token_dims=[64, 128, 256, 512], downsample_ratios=[4, 2, 2, 2],
                            NC_depth=[2, 2, 8, 2], NC_heads=[1, 2, 4, 8], RC_heads=[1, 1, 2, 4], mlp_ratio=4., NC_group=[1, 32, 64, 128], RC_group=[1, 16, 32, 64], window_size=7, **kwargs)
    model.default_cfg = default_cfgs['ViTAE_stages3_7']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def ViTAE_DeformWindow_NoShift_RPE_1234_basic_stages4_14(pretrained=False, **kwargs): # adopt performer for tokens to token
    # if pretrained:
        # kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = ViTAE_DeformWindow_NoShift_basic(RC_tokens_type=['VSA', 'VSA', 'VSA', 'VSA'], NC_tokens_type=['VSA', 'VSA', 'VSA', 'VSA'], stages=4, embed_dims=[64, 64, 128, 256], token_dims=[64, 128, 256, 512], downsample_ratios=[4, 2, 2, 2],
                            NC_depth=[2, 2, 8, 2], NC_heads=[1, 2, 4, 8], RC_heads=[1, 1, 2, 4], mlp_ratio=4., NC_group=[1, 32, 64, 128], RC_group=[1, 16, 32, 64], window_size=7, relative_pos=True, **kwargs)
    model.default_cfg = default_cfgs['ViTAE_stages3_7']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def ViTAE_DeformWindow_NoShift_CPE_1234_basic_stages4_14(pretrained=False, **kwargs): # adopt performer for tokens to token
    # if pretrained:
        # kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = ViTAE_DeformWindow_NoShift_basic(RC_tokens_type=['VSA', 'VSA', 'VSA', 'VSA'], NC_tokens_type=['VSA', 'VSA', 'VSA', 'VSA'], stages=4, embed_dims=[64, 64, 128, 256], token_dims=[64, 128, 256, 512], downsample_ratios=[4, 2, 2, 2],
                            NC_depth=[2, 2, 8, 2], NC_heads=[1, 2, 4, 8], RC_heads=[1, 1, 2, 4], mlp_ratio=4., NC_group=[1, 32, 64, 128], RC_group=[1, 16, 32, 64], window_size=7, cpe=True, **kwargs)
    model.default_cfg = default_cfgs['ViTAE_stages3_7']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def ViTAE_DeformWindow_NoShift_CPE_1234_head2_basic_stages4_14(pretrained=False, **kwargs): # adopt performer for tokens to token
    # if pretrained:
        # kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = ViTAE_DeformWindow_NoShift_basic(RC_tokens_type=['VSA', 'VSA', 'VSA', 'VSA'], NC_tokens_type=['VSA', 'VSA', 'VSA', 'VSA'], stages=4, embed_dims=[64, 64, 128, 256], token_dims=[64, 128, 256, 512], downsample_ratios=[4, 2, 2, 2],
                            NC_depth=[2, 2, 8, 2], NC_heads=[2, 4, 8, 16], RC_heads=[1, 1, 2, 4], mlp_ratio=4., NC_group=[1, 32, 64, 128], RC_group=[1, 16, 32, 64], window_size=7, cpe=True, **kwargs)
    model.default_cfg = default_cfgs['ViTAE_stages3_7']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def ViTAE_DeformWindow_NoShift_CPE_1234_NChead2_RChead2_basic_stages4_14(pretrained=False, **kwargs): # adopt performer for tokens to token
    # if pretrained:
        # kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = ViTAE_DeformWindow_NoShift_basic(RC_tokens_type=['VSA', 'VSA', 'VSA', 'VSA'], NC_tokens_type=['VSA', 'VSA', 'VSA', 'VSA'], stages=4, embed_dims=[64, 64, 128, 256], token_dims=[64, 128, 256, 512], downsample_ratios=[4, 2, 2, 2],
                            NC_depth=[2, 2, 8, 2], NC_heads=[2, 4, 8, 16], RC_heads=[2, 2, 4, 8], mlp_ratio=4., NC_group=[1, 32, 64, 128], RC_group=[1, 16, 32, 64], window_size=7, cpe=True, **kwargs)
    model.default_cfg = default_cfgs['ViTAE_stages3_7']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def ViTAE_DeformWindow_NoShift_CPE_RPE_1234_NChead2_RChead2_basic_stages4_14(pretrained=False, **kwargs): # adopt performer for tokens to token
    # if pretrained:
        # kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = ViTAE_DeformWindow_NoShift_basic(RC_tokens_type=['VSA', 'VSA', 'VSA', 'VSA'], NC_tokens_type=['VSA', 'VSA', 'VSA', 'VSA'], stages=4, embed_dims=[64, 64, 128, 256], token_dims=[64, 128, 256, 512], downsample_ratios=[4, 2, 2, 2],
                            NC_depth=[2, 2, 8, 2], NC_heads=[2, 4, 8, 16], RC_heads=[2, 2, 4, 8], mlp_ratio=4., NC_group=[1, 32, 64, 128], RC_group=[1, 16, 32, 64], window_size=7, cpe=True, relative_pos=True, **kwargs)
    model.default_cfg = default_cfgs['ViTAE_stages3_7']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model
# @register_model
# def ViTAE_Window_NoShift_12_basic_stages3_7(pretrained=False, **kwargs): # adopt performer for tokens to token
#     # if pretrained:
#         # kwargs.setdefault('qk_scale', 256 ** -0.5)
#     model = ViTAE_Window_NoShift_basic(RC_tokens_type=['swin', 'swin', 'transformer'], NC_tokens_type=['swin', 'swin', 'transformer'], stages=3, embed_dims=[32, 64, 128], token_dims=[64, 128, 256], downsample_ratios=[4, 2, 2],
#                             NC_depth=[1, 1, 5], NC_heads=[1, 2, 4], RC_heads=[1, 2, 4], mlp_ratio=2., NC_group=[1, 16, 32], RC_group=[1, 16, 32], **kwargs)
#     model.default_cfg = default_cfgs['ViTAE_stages3_7']
#     if pretrained:
#         load_pretrained(
#             model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
#     return model

# @register_model
# def ViTAE_Window_NoShift_1_basic_stages4_14(pretrained=False, **kwargs): # adopt performer for tokens to token
#     # if pretrained:
#         # kwargs.setdefault('qk_scale', 256 ** -0.5)
#     model = ViTAE_Window_NoShift_basic(RC_tokens_type=['swin', 'transformer', 'transformer', 'transformer'], NC_tokens_type=['swin', 'transformer', 'transformer', 'transformer'], stages=4, embed_dims=[64, 64, 128, 256], token_dims=[64, 128, 256, 512], downsample_ratios=[4, 2, 2, 2],
#                             NC_depth=[2, 2, 8, 2], NC_heads=[1, 2, 4, 8], RC_heads=[1, 1, 2, 4], mlp_ratio=4., NC_group=[1, 32, 64, 128], RC_group=[1, 16, 32, 64], window_size=7, **kwargs)
#     model.default_cfg = default_cfgs['ViTAE_stages3_7']
#     if pretrained:
#         load_pretrained(
#             model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
#     return model

# @register_model
# def ViTAE_Window_NoShift_12_basic_stages4_14(pretrained=False, **kwargs): # adopt performer for tokens to token
#     # if pretrained:
#         # kwargs.setdefault('qk_scale', 256 ** -0.5)
#     model = ViTAE_Window_NoShift_basic(RC_tokens_type=['swin', 'swin', 'transformer', 'transformer'], NC_tokens_type=['swin', 'swin', 'transformer', 'transformer'], stages=4, embed_dims=[64, 64, 128, 256], token_dims=[64, 128, 256, 512], downsample_ratios=[4, 2, 2, 2],
#                             NC_depth=[2, 2, 8, 2], NC_heads=[1, 2, 4, 8], RC_heads=[1, 1, 2, 4], mlp_ratio=4., NC_group=[1, 32, 64, 128], RC_group=[1, 16, 32, 64], window_size=7, **kwargs)
#     model.default_cfg = default_cfgs['ViTAE_stages3_7']
#     if pretrained:
#         load_pretrained(
#             model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
#     return model

# @register_model
# def ViTAE_Window_NoShift_123_basic_stages4_14(pretrained=False, **kwargs): # adopt performer for tokens to token
#     # if pretrained:
#         # kwargs.setdefault('qk_scale', 256 ** -0.5)
#     model = ViTAE_Window_NoShift_basic(RC_tokens_type=['swin', 'swin', 'swin', 'transformer'], NC_tokens_type=['swin', 'swin', 'swin', 'transformer'], stages=4, embed_dims=[64, 64, 128, 256], token_dims=[64, 128, 256, 512], downsample_ratios=[4, 2, 2, 2],
#                             NC_depth=[2, 2, 8, 2], NC_heads=[1, 2, 4, 8], RC_heads=[1, 1, 2, 4], mlp_ratio=4., NC_group=[1, 32, 64, 128], RC_group=[1, 16, 32, 64], window_size=7, **kwargs)
#     model.default_cfg = default_cfgs['ViTAE_stages3_7']
#     if pretrained:
#         load_pretrained(
#             model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
#     return model

# @register_model
# def ViTAE_Window_NoShift_1234_basic_stages4_14(pretrained=False, **kwargs): # adopt performer for tokens to token
#     # if pretrained:
#         # kwargs.setdefault('qk_scale', 256 ** -0.5)
#     model = ViTAE_Window_NoShift_basic(RC_tokens_type=['swin', 'swin', 'swin', 'swin'], NC_tokens_type=['swin', 'swin', 'swin', 'swin'], stages=4, embed_dims=[64, 64, 128, 256], token_dims=[64, 128, 256, 512], downsample_ratios=[4, 2, 2, 2],
#                             NC_depth=[2, 2, 8, 2], NC_heads=[1, 2, 4, 8], RC_heads=[1, 1, 2, 4], mlp_ratio=4., NC_group=[1, 32, 64, 128], RC_group=[1, 16, 32, 64], window_size=7, **kwargs)
#     model.default_cfg = default_cfgs['ViTAE_stages3_7']
#     if pretrained:
#         load_pretrained(
#             model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
#     return model

# @register_model
# def ViTAE_Window_NoShift_stem_12_basic_stages4_14(pretrained=False, **kwargs): # adopt performer for tokens to token
#     # if pretrained:
#         # kwargs.setdefault('qk_scale', 256 ** -0.5)
#     model = ViTAE_Window_NoShift_basic(RC_tokens_type=['stem', 'swin', 'transformer', 'transformer'], NC_tokens_type=['swin', 'swin', 'transformer', 'transformer'], stages=4, embed_dims=[64, 64, 128, 256], token_dims=[64, 128, 256, 512], downsample_ratios=[4, 2, 2, 2],
#                             NC_depth=[3, 2, 8, 2], NC_heads=[1, 2, 4, 8], RC_heads=[1, 1, 2, 4], mlp_ratio=4., NC_group=[1, 32, 64, 128], RC_group=[1, 16, 32, 64], window_size=7, **kwargs)
#     model.default_cfg = default_cfgs['ViTAE_stages3_7']
#     if pretrained:
#         load_pretrained(
#             model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
#     return model

# @register_model
# def ViTAE_Window_NoShift_12_basic_rpe_stages4_14(pretrained=False, **kwargs): # adopt performer for tokens to token
#     # if pretrained:
#         # kwargs.setdefault('qk_scale', 256 ** -0.5)
#     model = ViTAE_Window_NoShift_basic(RC_tokens_type=['swin', 'swin', 'transformer', 'transformer'], NC_tokens_type=['swin', 'swin', 'transformer', 'transformer'], stages=4, embed_dims=[64, 64, 128, 256], token_dims=[64, 128, 256, 512], downsample_ratios=[4, 2, 2, 2],
#                             NC_depth=[2, 2, 8, 2], NC_heads=[1, 2, 4, 8], RC_heads=[1, 1, 2, 4], mlp_ratio=4., NC_group=[1, 32, 64, 128], RC_group=[1, 16, 32, 64], window_size=7, relative_pos=True, **kwargs)
#     model.default_cfg = default_cfgs['ViTAE_stages3_7']
#     if pretrained:
#         load_pretrained(
#             model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
#     return model

# @register_model
# def ViTAE_Window_NoShift_stem_12_basic_rpe_stages4_14(pretrained=False, **kwargs): # adopt performer for tokens to token
#     # if pretrained:
#         # kwargs.setdefault('qk_scale', 256 ** -0.5)
#     model = ViTAE_Window_NoShift_basic(RC_tokens_type=['stem', 'swin', 'transformer', 'transformer'], NC_tokens_type=['swin', 'swin', 'transformer', 'transformer'], stages=4, embed_dims=[64, 64, 128, 256], token_dims=[64, 128, 256, 512], downsample_ratios=[4, 2, 2, 2],
#                             NC_depth=[3, 2, 8, 2], NC_heads=[1, 2, 4, 8], RC_heads=[1, 1, 2, 4], mlp_ratio=4., NC_group=[1, 32, 64, 128], RC_group=[1, 16, 32, 64], window_size=7, relative_pos=True, **kwargs)
#     model.default_cfg = default_cfgs['ViTAE_stages3_7']
#     if pretrained:
#         load_pretrained(
#             model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
#     return model

# @register_model
# def ViTAE_Window_NoShift_basic_stages4_14(pretrained=False, **kwargs): # adopt performer for tokens to token
#     # if pretrained:
#         # kwargs.setdefault('qk_scale', 256 ** -0.5)
#     model = ViTAE_Window_NoShift_basic(RC_tokens_type=['swin', 'swin', 'swin', 'swin'], NC_tokens_type=['swin', 'swin', 'swin', 'swin'], stages=4, embed_dims=[64, 64, 128, 256], token_dims=[64, 128, 256, 512], downsample_ratios=[4, 2, 2, 2],
#                             NC_depth=[2, 2, 8, 2], NC_heads=[1, 2, 4, 8], RC_heads=[1, 1, 2, 4], mlp_ratio=4., NC_group=[1, 32, 64, 128], RC_group=[1, 16, 32, 64], window_size=7, **kwargs)
#     model.default_cfg = default_cfgs['ViTAE_stages3_7']
#     if pretrained:
#         load_pretrained(
#             model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
#     return model

# @register_model
# def ViTAE_Window_NoShift_12_basic_stages4_17(pretrained=False, **kwargs): # adopt performer for tokens to token
#     # if pretrained:
#         # kwargs.setdefault('qk_scale', 256 ** -0.5)
#     model = ViTAE_Window_NoShift_basic(RC_tokens_type=['swin', 'swin', 'transformer', 'transformer'], NC_tokens_type=['swin', 'swin', 'transformer', 'transformer'], stages=4, embed_dims=[64, 64, 192, 384], token_dims=[96, 192, 384, 768], downsample_ratios=[4, 2, 2, 2],
#                             NC_depth=[2, 2, 11, 2], NC_heads=[1, 2, 4, 8], RC_heads=[1, 1, 2, 4], mlp_ratio=4., NC_group=[1, 32, 64, 128], RC_group=[1, 16, 32, 64], **kwargs)
#     model.default_cfg = default_cfgs['ViTAE_stages3_7']
#     if pretrained:
#         load_pretrained(
#             model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
#     return model

# @register_model
# def ViTAE_Window_NoShift_12_basic_stages4_18(pretrained=False, **kwargs): # adopt performer for tokens to token
#     # if pretrained:
#         # kwargs.setdefault('qk_scale', 256 ** -0.5)
#     model = ViTAE_Window_NoShift_basic(RC_tokens_type=['swin', 'swin', 'transformer', 'transformer'], NC_tokens_type=['swin', 'swin', 'transformer', 'transformer'], stages=4, embed_dims=[96, 96, 256, 512], token_dims=[128, 256, 512, 1024], downsample_ratios=[4, 2, 2, 2],
#                             NC_depth=[2, 2, 12, 2], NC_heads=[1, 4, 8, 16], RC_heads=[1, 1, 4, 8], mlp_ratio=4., NC_group=[1, 32, 64, 128], RC_group=[1, 16, 32, 64], **kwargs)
#     model.default_cfg = default_cfgs['ViTAE_stages3_7']
#     if pretrained:
#         load_pretrained(
#             model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
#     return model


# @register_model
# def ViTAE_Window_NoShift_basic_stages4_17(pretrained=False, **kwargs): # adopt performer for tokens to token
#     # if pretrained:
#         # kwargs.setdefault('qk_scale', 256 ** -0.5)
#     model = ViTAE_Window_NoShift_basic(RC_tokens_type=['swin', 'swin', 'swin', 'swin'], NC_tokens_type=['swin', 'swin', 'swin', 'swin'], stages=4, embed_dims=[64, 64, 192, 384], token_dims=[96, 192, 384, 768], downsample_ratios=[4, 2, 2, 2],
#                             NC_depth=[2, 2, 11, 2], NC_heads=[1, 2, 4, 8], RC_heads=[1, 1, 2, 4], mlp_ratio=4., NC_group=[1, 32, 64, 128], RC_group=[1, 16, 32, 64], **kwargs)
#     model.default_cfg = default_cfgs['ViTAE_stages3_7']
#     if pretrained:
#         load_pretrained(
#             model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
#     return model