import torch
import torch.nn as nn
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

class FlashAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(0, 2, 3, 1, 4)  # [B, 3, num_heads, N, head_dim]
        
        # Flash attention expects packed qkv
        output = flash_attn_qkvpacked_func(qkv, dropout_p=self.dropout if self.training else 0.0)
        
        output = output.reshape(B, N, C)
        output = self.proj(output)
        return output 