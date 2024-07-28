import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F

from encoder.multiheadattention import MultiHeadAttention, LayerNormalization, PositionwiseFeedForward
from encoder.encoding import SentenceEmbedding

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, self_attention_mask):
        attention_output = self.attention(x, mask=self_attention_mask)
        attention_output = self.dropout1(attention_output)
        x = self.norm1(x + attention_output)
        
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output)
        x = self.norm2(x + ffn_output)
        
        return x

class Encoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 vocab_size):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, vocab_size)
        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                      for _ in range(num_layers)])

    def forward(self, x, self_attention_mask=None):
        #print(f"Encoder input shape: {x.shape}")
        x = self.sentence_embedding(x)
        #print(f"After sentence embedding: {x.shape}")
        x = self.layers(x, self_attention_mask)
        #print(f"Encoder output shape: {x.shape}")
        return x

class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x, self_attention_mask  = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x