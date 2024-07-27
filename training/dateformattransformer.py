import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F


from encoder.encoderlayer import Encoder
from decoder.decoderlayer import Decoder

def create_masks(src, tgt, pad_idx):
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    
    if tgt is not None:
        tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(src.device)
        tgt_mask = tgt_mask & nopeak_mask
    else:
        tgt_mask = None
    
    return src_mask, tgt_mask

class DateFormatTransformer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, input_vocab_size, output_vocab_size, pad_idx):
        super().__init__()
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, input_vocab_size)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, output_vocab_size)
        self.linear = nn.Linear(d_model, output_vocab_size)
        self.max_sequence_length = max_sequence_length
        self.output_vocab_size = output_vocab_size
        self.pad_idx = pad_idx

    def forward(self, src, tgt=None):
        device = src.device
        src_mask, tgt_mask = create_masks(src, tgt, self.pad_idx) if tgt is not None else (None, None)
        
        encoder_output = self.encoder(src, src_mask)
        
        if tgt is None:
            # Inference mode
            output = torch.zeros((src.size(0), 1, self.output_vocab_size), device=device)
            for _ in range(self.max_sequence_length):
                tgt_mask = create_masks(src, output.argmax(dim=-1), self.pad_idx)[1]
                decoder_output = self.decoder(encoder_output, output.argmax(dim=-1), tgt_mask, src_mask)
                step_output = self.linear(decoder_output[:, -1:, :])
                output = torch.cat([output, step_output], dim=1)
                if output.argmax(dim=-1)[:, -1].item() == 2:  # Assuming 2 is the <END> token
                    break
            return output[:, 1:, :]  # Remove the initial zero
        else:
            # Training mode
            decoder_output = self.decoder(encoder_output, tgt, tgt_mask, src_mask)
            return self.linear(decoder_output)