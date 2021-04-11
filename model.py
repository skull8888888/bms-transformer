import torch
import torch.nn as nn
import pytorch_lightning as pl

import numpy as np
from utils import get_score
from config import CFG
from encoder import Encoder
from decoder import Decoder

from config import CFG

from dataset import generate_square_subsequent_mask

class Model(pl.LightningModule):
    
    def __init__(self, tokenizer, train_loader_len=100):
        super(Model, self).__init__()    
        
        self.tokenizer = tokenizer
        
        self.train_loader_len=train_loader_len
        
        self.encoder = Encoder(CFG.d_model, CFG.n_head, CFG.encoder_layers)
        self.decoder = Decoder(len(tokenizer), CFG.max_len, CFG.d_model, CFG.n_head, CFG.decoder_layers)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.stoi["<pad>"])
    
    def predict(self, imgs, decode_lengths):
        
        enc_output = self.encoder(imgs)

        batch_size = enc_output.size(0)
        encoder_dim = enc_output.size(-1)
        
        # embed start token 
        input_tokens = torch.ones(batch_size).type_as(enc_output).long() * self.tokenizer.stoi["<sos>"]
        input_tokens = input_tokens.unsqueeze(1)
        
        predictions = []
        
        end_condition = torch.zeros(batch_size, dtype=torch.long).type_as(enc_output)
    
        # predict sequence        
        for t in range(decode_lengths):
            
            x = self.decoder(input_tokens, enc_output)

            preds = x.squeeze(1)

            output_token = torch.argmax(preds, -1)

            predictions.append(output_token)
                        
            end_condition = torch.logical_or(end_condition, (output_token == self.tokenizer.stoi["<eos>"]))
            if end_condition.sum() == batch_size:
                break
            output_token = output_token.unsqueeze(1)
            input_tokens = torch.cat([input_tokens, output_token], dim=1)
        
        predictions = torch.stack(predictions,dim=-1)

        return predictions
        
    def training_step(self, batch, batch_nb):
        
        imgs, labels, label_lengths, mask, pad_masks = batch
        

        enc_output = self.encoder(imgs)
        
        pred = self.decoder(labels, enc_output, mask, pad_masks)
        pred = pred.view(-1,pred.size(-1))
        
        pads = torch.full((labels.size(0),1), self.tokenizer.stoi["<pad>"]).type_as(labels)
        labels = torch.cat([labels[:,1:], pads], dim=-1).flatten()
          
        loss = self.criterion(pred, labels)
        
        lr = self.scheduler.get_last_lr()[0]
        self.log('lr', lr, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_nb):
        
        imgs, labels, label_lengths, masks, pad_masks = batch

        max_len = max(label_lengths)
        
        pred = self.predict(imgs, max_len)
        pred = pred.detach().cpu().numpy()
        
        labels = labels.detach().cpu().numpy()
        
        pred_inchi = self.tokenizer.predict_captions(pred)
        target_inchi = self.tokenizer.predict_captions(labels[:,1:])
        
        return {'pred': pred_inchi, 'target': target_inchi}
    
    
    def validation_epoch_end(self, outputs):
        
        pred = np.concatenate([out['pred'] for out in outputs])
        target = np.concatenate([out['target'] for out in outputs])

        with open("pred.txt","w") as f:
            for s in pred:
                f.write(s + '\n')
        score = get_score(target, pred)
        
        self.log('LD', score, prog_bar=True)
                
    
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None, on_tpu=False, using_native_amp=True, using_lbfgs=False):
                
        optimizer.step(closure=second_order_closure)
        self.scheduler.step()
#         self.scheduler.step(current_epoch + batch_nb / self.train_loader_len)
        
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=CFG.lr, weight_decay=CFG.l2) 
         
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            div_factor=CFG.div_factor,
            pct_start=CFG.pct_start,
            max_lr=CFG.lr,
            anneal_strategy='cos',
            steps_per_epoch=self.train_loader_len, 
            epochs=CFG.epochs)
        
#         self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG.epochs, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
        
        return optimizer
#         lr_scheduler = {'scheduler': torch.optim.lr_scheduler.OneCycleLR(
#                             optimizer, 
#                             div_factor=25,
#                             pct_start=0.3,
#                             max_lr=CFG.lr,
#                             anneal_strategy='cos',
#                             steps_per_epoch=self.train_loader_len, 
#                             epochs=CFG.epochs),
#                         'name': 'learning_rate',
#                         'interval':'step',
#                         'frequency': 1}
    
#         return [optimizer], [lr_scheduler]

#      def configure_optimizers(self):
#             self.optimizer = optim.AdamW(self.model.parameters(), self.args.learning_rate)

#             ds_len = self.train_dataset.__len__()
#             total_bs = self.args.batch_size*self.args.gpus
#             self.scheduler = optim.lr_scheduler.OneCycleLR(
#                                             self.optimizer, max_lr=self.args.learning_rate,
#                                             anneal_strategy='linear', div_factor=100,
#                                             steps_per_epoch=int((ds_len/total_bs)),
#                                             epochs=self.args.epochs)

#             sched = {
#                 'scheduler': self.scheduler,
#                 'interval': 'step',
#             }
#             return [self.optimizer], [sched]