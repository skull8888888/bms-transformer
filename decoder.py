import torch
import torch.nn as nn
import math
import numpy as np

from attention import ResidualDecoderAttentionBlock

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, d_model: int, n_head: int, layers: int):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.layers = layers
        self.d_model = d_model
        
        self.transformer = nn.ModuleList([ResidualDecoderAttentionBlock(d_model, n_head) for _ in range(layers)])
        
        self.embedding = nn.Embedding(vocab_size, d_model) 
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(p=0.1)
        
        self.ln_post = nn.LayerNorm(d_model)
        
        self.initialize_parameters()
        
    def initialize_parameters(self):
        
        nn.init.normal_(self.embedding.weight, std=0.02)

        proj_std = (self.d_model ** -0.5) * ((2 * self.layers) ** -0.5)
        attn_std = self.d_model ** -0.5
        fc_std = (2 * self.d_model) ** -0.5
        
        for block in self.transformer:

            nn.init.normal_(block.attn_1.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn_1.out_proj.weight, std=proj_std)

            nn.init.normal_(block.attn_2.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn_2.out_proj.weight, std=proj_std)
            
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def forward(self, tokens, enc_output, memory=None, mask=None, pad_mask=None):
        """
        dec_input (batch, seq, 1)
        enc_output (batch, layers, seq, features)

        """
        
        enc_output = enc_output.permute(1, 0, 2)  # NLD -> LND
        
        x = self.embedding(tokens) #/ math.sqrt(self.d_model)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.pos_enc(x)
                
        new_memory = []
            
        for i, att_block in enumerate(self.transformer):

            if i != 0:

                if memory is not None:
                    x = torch.cat([memory[i-1], x])
                new_memory.append(x)

            x = att_block(x, enc_output, mask, pad_mask)        
            
            
        x = self.ln_post(x)
        
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        
        x = self.fc(self.dropout(x))
        
        if not self.training:

            if self.layers > 0:
                new_memory = torch.stack(new_memory)
            else:
                new_memory = None

            return x, new_memory
            
        return x
    