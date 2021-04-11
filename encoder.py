import torch 
import torch.nn as nn
import timm
import math
from attention import ResidualEncoderAttentionBlock
from config import CFG

class EncoderPositionalEncoding(nn.Module):
    def __init__(self, d_model, height, width):
        super().__init__()

        
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        self.d_model = d_model
    
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        
        self.register_buffer('pe', pe)
        
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        """
        x: (batch_size, d_model, width*height)
        """
        x = x + self.pe.view(self.d_model, -1)
        x = self.dropout(x)
        
        return x

    
class FE(nn.Module):
    def __init__(self, d_model, dropout=0.5):
        super().__init__()
        
        self.d_model = d_model

        resnet = timm.create_model(CFG.backbone, pretrained=False)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        
        seq_l = 7
        
        self.d_encoder = CFG.d_encoder
        
        self.pos_enc = EncoderPositionalEncoding(d_model,seq_l,seq_l)
        self.fc = nn.Linear(self.d_encoder, d_model, bias=False)
        
    def forward(self, x):
        
        batch_size = x.size(0)
        
        x = self.resnet(x) # (batch, d_encoder, height, width)
        
        x = x.view(batch_size, self.d_encoder, -1) # (batch, d_encoder, height * width)

        x = x.permute(0,2,1) # (batch, height*width, d_encoder)
        x = self.fc(x)
        x = x.permute(0,2,1) # (batch, d_model, height*width)
        
        x = self.pos_enc(x)
        
        x = x.permute(0,2,1) # (batch, height*width, d_model)

        return x
    
    
class Encoder(nn.Module):
    def __init__(self, d_model: int, n_head: int, layers: int):
        super().__init__()
        
        self.layers = layers
        
        self.transformer = nn.Sequential(*[ResidualEncoderAttentionBlock(d_model, n_head) for _ in range(layers)])
#         self.transformer = ResidualEncoderAttentionBlock(d_model, n_head)

        self.fe = FE(d_model)
        

    def forward(self, x):
        """
        x: (batch, c, h, w)
        
        output: (batch, seq, d_model)

        """
        x = self.fe(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2) # LND -> NLD
            
        return x
    