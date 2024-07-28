import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled = scaled.masked_fill(mask == 0, -1e9)
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = nn.Linear(d_model , 2 * d_model)
        self.q_layer = nn.Linear(d_model , d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, y, mask):
        batch_size, x_seq_length, _ = x.size()
        _, y_seq_length, _ = y.size()
        #print(f"MultiHeadCrossAttention input shapes: x={x.shape}, y={y.shape}")
        
        kv = self.kv_layer(x)
        q = self.q_layer(y)
        
        #print(f"After linear layers: kv={kv.shape}, q={q.shape}")
        
        kv = kv.reshape(batch_size, x_seq_length, self.num_heads, 2 * self.head_dim)
        q = q.reshape(batch_size, y_seq_length, self.num_heads, self.head_dim)
        
        #print(f"After reshape: kv={kv.shape}, q={q.shape}")
        
        kv = kv.permute(0, 2, 1, 3)
        q = q.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)
        
        #print(f"Before scaled_dot_product: q={q.shape}, k={k.shape}, v={v.shape}")
        
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, y_seq_length, self.d_model)
        out = self.linear_layer(values)
        return out
