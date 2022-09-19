# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
import torch.nn.functional as F

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class VSAWindowAttention(nn.Module):
    def __init__(self, dim, num_heads, out_dim=None, window_size=1, qkv_bias=True, qk_scale=None, 
            attn_drop=0., proj_drop=0.,
            img_size=(1,1),):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim or dim
        self.relative_pos_embedding = True
        head_dim = dim // self.num_heads
        self.ws = window_size
        self.shift_size = 0
        
        self.padding_bottom = (self.ws - self.img_size[0] % self.ws) % self.ws
        self.padding_right = (self.ws - self.img_size[1] % self.ws) % self.ws

        self.sampling_offsets = nn.Sequential(
            nn.AvgPool2d(kernel_size=window_size, stride=window_size),
            nn.LeakyReLU(), 
            nn.Conv2d(dim, self.num_heads * 2, kernel_size=1, stride=1)
        )
        self.sampling_scales = nn.Sequential(
            nn.AvgPool2d(kernel_size=window_size, stride=window_size), 
            nn.LeakyReLU(), 
            nn.Conv2d(dim, self.num_heads * 2, kernel_size=1, stride=1)
        )

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, out_dim * 3, 1, bias=qkv_bias)
        # self.kv = nn.Conv2d(dim, dim*2, 1, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(out_dim, out_dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((window_size + window_size - 1) * (window_size + window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)
            print('The relative_pos_embedding is used')

        h, w = self.img_size
        h, w = h + self.shift_size + self.padding_bottom, w + self.shift_size + self.padding_right
        image_reference_w = torch.linspace(-1, 1, w)
        image_reference_h = torch.linspace(-1, 1, h)
        image_reference = torch.stack(torch.meshgrid(image_reference_w, image_reference_h), 0).permute(0, 2, 1).unsqueeze(0) # 2, h, w
        window_reference = nn.functional.avg_pool2d(image_reference, kernel_size=self.ws)
        window_num_h, window_num_w = window_reference.shape[-2:]
        window_reference = window_reference.reshape(1, 2, window_num_h, 1, window_num_w, 1)

        base_coords_h = torch.arange(self.ws) * 2 * self.ws / self.ws / (h-1)
        base_coords_h = (base_coords_h - base_coords_h.mean())
        base_coords_w = torch.arange(self.ws) * 2 * self.ws / self.ws / (w-1)
        base_coords_w = (base_coords_w - base_coords_w.mean())

        expanded_base_coords_h = base_coords_h.unsqueeze(dim=0).repeat(window_num_h, 1)
        assert expanded_base_coords_h.shape[0] == window_num_h
        assert expanded_base_coords_h.shape[1] == self.ws
        expanded_base_coords_w = base_coords_w.unsqueeze(dim=0).repeat(window_num_w, 1)
        assert expanded_base_coords_w.shape[0] == window_num_w
        assert expanded_base_coords_w.shape[1] == self.ws
        expanded_base_coords_h = expanded_base_coords_h.reshape(-1)
        expanded_base_coords_w = expanded_base_coords_w.reshape(-1)
        coords = torch.stack(torch.meshgrid(expanded_base_coords_w, expanded_base_coords_h), 0).permute(0, 2, 1).reshape(1, 2, window_num_h, self.ws, window_num_w, self.ws)
        self.register_buffer('base_coords', window_reference+coords)
        self.register_buffer('coords', coords)

    def forward(self, x):
        b, _, h, w = x.shape
        shortcut = x
        assert h == self.img_size[0]
        assert w == self.img_size[1]

        x = torch.nn.functional.pad(x, (self.shift_size, self.padding_right, self.shift_size, self.padding_bottom))
        window_num_h, window_num_w = self.base_coords.shape[-4], self.base_coords.shape[-2]

        coords = self.base_coords.repeat(b*self.num_heads, 1, 1, 1, 1, 1)
        sampling_offsets = self.sampling_offsets(x)
        num_predict_total = b * self.num_heads
        sampling_offsets = sampling_offsets.reshape(num_predict_total, 2, window_num_h, window_num_w)
        sampling_offsets[:, 0, ...] = sampling_offsets[:, 0, ...] / (w // self.ws)
        sampling_offsets[:, 1, ...] = sampling_offsets[:, 1, ...] / (h // self.ws)
        
        sampling_scales = self.sampling_scales(x)       #B, heads*2, h // window_size, w // window_size
        sampling_scales = sampling_scales.reshape(num_predict_total, 2, window_num_h, window_num_w)
        
        coords = coords + self.coords * sampling_scales[:, :, :, None, :, None] + sampling_offsets[:, :, :, None, :, None]
        sample_coords = coords.permute(0, 2, 3, 4, 5, 1).reshape(num_predict_total, self.ws*window_num_h, self.ws*window_num_w, 2)


        qkv = self.qkv(shortcut).reshape(b, 3, self.num_heads, self.out_dim // self.num_heads, h, w).transpose(1, 0).reshape(3*b*self.num_heads, self.out_dim // self.num_heads, h, w)
        qkv = torch.nn.functional.pad(qkv, (self.shift_size, self.padding_right, self.shift_size, self.padding_bottom)).reshape(3, b*self.num_heads, self.out_dim // self.num_heads, h+self.shift_size+self.padding_bottom, w+self.shift_size+self.padding_right)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_selected = F.grid_sample(
                        k.reshape(num_predict_total, self.out_dim // self.num_heads, h+self.shift_size+self.padding_bottom, w+self.shift_size+self.padding_right), 
                        grid=sample_coords, padding_mode='zeros', align_corners=True
                        ).reshape(b*self.num_heads, self.out_dim // self.num_heads, h+self.shift_size+self.padding_bottom, w+self.shift_size+self.padding_right)
        v_selected = F.grid_sample(
                        v.reshape(num_predict_total, self.out_dim // self.num_heads, h+self.shift_size+self.padding_bottom, w+self.shift_size+self.padding_right), 
                        grid=sample_coords, padding_mode='zeros', align_corners=True
                        ).reshape(b*self.num_heads, self.out_dim // self.num_heads, h+self.shift_size+self.padding_bottom, w+self.shift_size+self.padding_right)

        q = q.reshape(b, self.num_heads, self.out_dim//self.num_heads, window_num_h, self.ws, window_num_w, self.ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b*window_num_h*window_num_w, self.num_heads, self.ws*self.ws, self.out_dim//self.num_heads)
        k = k_selected.reshape(b, self.num_heads, self.out_dim//self.num_heads, window_num_h, self.ws, window_num_w, self.ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b*window_num_h*window_num_w, self.num_heads, self.ws*self.ws, self.out_dim//self.num_heads)
        v = v_selected.reshape(b, self.num_heads, self.out_dim//self.num_heads, window_num_h, self.ws, window_num_w, self.ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b*window_num_h*window_num_w, self.num_heads, self.ws*self.ws, self.out_dim//self.num_heads)
        
        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        out = attn @ v

        out = rearrange(out, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads, b=b, hh=window_num_h, ww=window_num_w, ws1=self.ws, ws2=self.ws)
        out = out[:, :, self.shift_size:h+self.shift_size, self.shift_size:w+self.shift_size]
 
        out = self.proj(out)
        out = self.proj_drop(out)

        return out
    
    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets[-1].weight, 0.)
        nn.init.constant_(self.sampling_offsets[-1].bias, 0.)
        nn.init.constant_(self.sampling_scales[-1].weight, 0.)
        nn.init.constant_(self.sampling_scales[-1].bias, 0.)

    def flops(self, ):
        N = self.ws * self.ws
        M = self.ws * self.ws
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * M
        #  x = (attn @ v)
        flops += self.num_heads * N * M * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        h, w = self.img_size[0] + self.shift_size + self.padding_bottom, self.img_size[1] + self.shift_size + self.padding_right
        flops *= (h / self.ws * w / self.ws)

        # for sampling
        flops_sampling = 0
        # pooling
        flops_sampling += h * w * self.dim
        # regressing the shift and scale
        flops_sampling += 2 * (h/self.ws + w/self.ws) * self.num_heads*2 * self.dim
        # calculating the coords
        flops_sampling += h/self.ws * self.ws * w/self.ws * self.ws * 2
        # grid sampling attended features
        flops_sampling += h/self.ws * self.ws * w/self.ws * self.ws * self.dim
        
        flops += flops_sampling

        return flops
