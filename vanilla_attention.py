# The following is the implementation of vanilla attention in python. 
# the goal is to gauge the thorughput of the vanilla attention implementation using pytorch 

import torch 
import math
import time 
import torch.nn.functional as F


def scaled_dot_product(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor
): 
    """
    q, k, v: tensors of shape (batch_size, num_heads, seq_len, head_dim)
    mask: broadcastable to (B,num_heads, seq_len, seq_len)

    Return: 
        output: (batch_size, num_heads, seq_len, head_dim)
        attn_weights: (batch_size, num_heads, seq_len, seq_len)
    """

    d_k = q.size(-1)
    scores = torch.matmul(q,k.transpose(-2,-1)) / math.sqrt(d_k)
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)
    return output, attn

def warmup(): 
    q_warmup = torch.randn(1, 32, 512, 64, device="cuda")
    k_warmup = torch.randn(1, 32, 512, 64, device="cuda")
    v_warmup = torch.randn(1, 32, 512, 64, device="cuda")
    for _ in range(3): 
        _ = scaled_dot_product(q_warmup, k_warmup, v_warmup)
    torch.cuda.synchronize() 

def main(): 
    batch_size: int = 1
    seq_len: int = 16384 
    hidden_dim: int = 2048
    num_heads: int = 32 # 16 if head_dim is 128 
    head_dim: int = 64 # 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    warmup()

    start = time.perf_counter_ns()
    output, attn = scaled_dot_product(q, k, v)
    end = time.perf_counter_ns()
    duration = (end - start) / 1e9 

    gflops = 4 * batch_size * num_heads * (seq_len**2) * head_dim / duration 

    print("gflops: ",gflops)

if __name__ == "__main__":
    main()
    