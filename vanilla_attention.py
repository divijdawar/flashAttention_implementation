# The following is the implementation of vanilla attention in python. 
# the goal is to gauge the thorughput of the vanilla attention implementation using pytorch 

import torch 
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def causal_mask(
    seq_len: int, 
    device: Optional[torch.device] = None
): 
    pass 

def scaled_dot_product(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    mask: Optional[torch.Tensor] = None, 
    dropout: Optional[torch.nn.Module] = None
): 
    """
    q, k, v: tensors of shape (batch_size, num_heads, seq_len, head_dim)
    mask: broadcastable to (B,num_heads, seq_len, seq_len)

    Return: 
        output: (batch_size, num_heads, seq_len, head_dim)
        attn_weights: (batch_size, num_heads, seq_len, seq_len)
    """

    pass 

class VanillaAttention():
    pass 

def warmup(): 
    pass 


if __name__ == "__main__":
    batch_size: int = 1
    seq_len: int = 16384 
    hidden_dim: int = 2048
    num_heads: int = 32 # 16 if head_dim is 128 
    head_dim: int = 64 # 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
