"""
T2T-ViT
"""
import torch
import torch.nn as nn

from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
import numpy as np
from .base_model import ViTAE_VSA_basic

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
    'ViTAE_VSA': _cfg(),
}

@register_model
def ViTAEv2_VSA_S(pretrained=False, **kwargs): # adopt performer for tokens to token
    # if pretrained:
        # kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = ViTAE_VSA_basic(RC_tokens_type=['VSA', 'VSA', 'VSA', 'VSA'], NC_tokens_type=['VSA', 'VSA', 'VSA', 'VSA'], stages=4, embed_dims=[64, 64, 128, 256], wide_pcm=False, token_dims=[64, 128, 256, 512], downsample_ratios=[4, 2, 2, 2],
                            NC_depth=[2, 2, 8, 2], NC_heads=[2, 4, 8, 16], RC_heads=[2, 2, 4, 8], mlp_ratio=4., NC_group=[1, 32, 64, 128], RC_group=[1, 16, 32, 64], window_size=7, cpe=True, **kwargs)
    model.default_cfg = default_cfgs['ViTAE_VSA']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def ViTAEv2_VSA_widePCM_S(pretrained=False, **kwargs): # adopt performer for tokens to token
    # if pretrained:
        # kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = ViTAE_VSA_basic(RC_tokens_type=['VSA', 'VSA', 'VSA', 'VSA'], NC_tokens_type=['VSA', 'VSA', 'VSA', 'VSA'], stages=4, embed_dims=[64, 64, 128, 256], wide_pcm=True, token_dims=[64, 128, 256, 512], downsample_ratios=[4, 2, 2, 2],
                            NC_depth=[2, 2, 8, 2], NC_heads=[2, 4, 8, 16], RC_heads=[2, 4, 8, 16], mlp_ratio=4., NC_group=[1, 32, 64, 128], RC_group=[1, 16, 32, 64], window_size=7, cpe=True, **kwargs)
    model.default_cfg = default_cfgs['ViTAE_VSA']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def ViTAEv2_VSA_widePCM_48M(pretrained=False, **kwargs): # adopt performer for tokens to token
    # if pretrained:
        # kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = ViTAE_VSA_basic(RC_tokens_type=['VSA', 'VSA', 'VSA', 'VSA'], NC_tokens_type=['VSA', 'VSA', 'VSA', 'VSA'], stages=4, embed_dims=[96, 96, 192, 384], wide_pcm=True, token_dims=[96, 192, 384, 768], downsample_ratios=[4, 2, 2, 2],
                            NC_depth=[2, 2, 11, 2], NC_heads=[3, 6, 12, 24], RC_heads=[3, 6, 12, 24], mlp_ratio=4., NC_group=[1, 32, 64, 128], RC_group=[1, 16, 32, 64], window_size=7, cpe=True, **kwargs)
    model.default_cfg = default_cfgs['ViTAE_VSA']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def ViTAEv2_VSA_widePCM_B(pretrained=False, **kwargs): # adopt performer for tokens to token
    # if pretrained:
        # kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = ViTAE_VSA_basic(RC_tokens_type=['VSA', 'VSA', 'VSA', 'VSA'], NC_tokens_type=['VSA', 'VSA', 'VSA', 'VSA'], stages=4, embed_dims=[128, 128, 256, 512], wide_pcm=True, token_dims=[128, 256, 512, 1024], downsample_ratios=[4, 2, 2, 2],
                            NC_depth=[2, 2, 12, 2], NC_heads=[4, 8, 16, 32], RC_heads=[4, 8, 16, 32], mlp_ratio=4., NC_group=[1, 32, 64, 128], RC_group=[1, 16, 32, 64], window_size=7, cpe=True, **kwargs)
    model.default_cfg = default_cfgs['ViTAE_VSA']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def ViTAEv2_VSA_widePCM_B_ws12(pretrained=False, **kwargs): # adopt performer for tokens to token
    # if pretrained:
        # kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = ViTAE_VSA_basic(RC_tokens_type=['VSA', 'VSA', 'VSA', 'VSA'], NC_tokens_type=['VSA', 'VSA', 'VSA', 'VSA'], stages=4, embed_dims=[128, 128, 256, 512], token_dims=[128, 256, 512, 1024], downsample_ratios=[4, 2, 2, 2],
                            NC_depth=[2, 2, 12, 2], NC_heads=[4, 8, 16, 32], RC_heads=[4, 8, 16, 32], mlp_ratio=4., NC_group=[1, 32, 64, 128], RC_group=[1, 16, 32, 64], window_size=12, cpe=True, relative_pos=True, **kwargs)
    model.default_cfg = default_cfgs['ViTAE_stages3_7']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model
