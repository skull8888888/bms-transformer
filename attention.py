import torch
import torch.nn as nn


from collections import OrderedDict

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualEncoderAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("dropout_1", nn.Dropout(dropout)),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
            ("dropout_2", nn.Dropout(dropout))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        
    def attention(self, x: torch.Tensor):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
#         x = x + self.attention(x)
        x = x + self.mlp(self.ln_2(x))
        return x
    
class ResidualDecoderAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout=0.1):
        super().__init__()

        self.attn_1 = nn.MultiheadAttention(d_model, n_head)
        self.attn_2 = nn.MultiheadAttention(d_model, n_head)
        
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ln_3 = nn.LayerNorm(d_model)
        
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("dropout_1", nn.Dropout(dropout)),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
            ("dropout_2", nn.Dropout(dropout))
        ]))
        
        
    def attention(self, x: torch.Tensor, mask=None, pad_mask=None):

        # Q, K, V
        # Q - decoder_input
        # K - encoder_input
        # V - encoder_input
        
        if not self.training: 
            return self.attn_1(x[-1:,:,:], x, x, need_weights=False, attn_mask=mask, key_padding_mask=pad_mask)[0]

        return self.attn_1(x, x, x, need_weights=False, attn_mask=mask, key_padding_mask=pad_mask)[0]

    def enc_dec_attention(self, dec_input, enc_output):
        # Q, K, V
        # Q - decoder_input
        # K - encoder_input
        # V - encoder_input
        return self.attn_2(dec_input, enc_output, enc_output, need_weights=False)[0]

    
    def forward(self, dec_input, enc_output, mask=None, pad_mask=None):
        
        # putting layer normalization inside residual block is very important 
        if not self.training:
            x = dec_input[-1:,:,:] + self.attention(self.ln_1(dec_input), mask, pad_mask)
        else:
            x = dec_input + self.attention(self.ln_1(dec_input), mask, pad_mask)
        x = x + self.enc_dec_attention(self.ln_2(x), enc_output)
        x = x + self.mlp(self.ln_3(x))

        return x
    
    
