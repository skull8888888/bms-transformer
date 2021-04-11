
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from model import Model

from dataset import TrainDataset, TestDataset, get_transforms
from torch.utils.data import DataLoader

from tokenizer import Tokenizer
tokenizer = torch.load('./tokenizer2.pth')
import random
import matplotlib.pyplot as plt
from config import CFG
from tqdm import tqdm
import cv2

def inference(test_loader, model, tokenizer, device):

    text_preds = []
    tk0 = tqdm(test_loader, total=len(test_loader))
    
    for images in tk0:
        images = images.to(device)
        with torch.no_grad():
            preds = model.predict(images, CFG.max_len)
    
        preds_inchi = tokenizer.predict_captions(preds.detach().cpu().numpy())
            
        text_preds.append(preds_inchi)
        
    text_preds = np.concatenate(text_preds)
    return text_preds


test_df = pd.read_csv('test.csv')

device = 'cuda:0'

test_dataset = TestDataset(test_df, transform=get_transforms(data='valid'))
test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, drop_last=False)

model = Model.load_from_checkpoint(checkpoint_path='lightning_logs/version_1/checkpoints/epoch=06-LD=7.9722.ckpt', tokenizer=tokenizer, strict=False)

model.eval()
model.to(device)

predictions = inference(test_loader, model, tokenizer, device)

sub_df = pd.read_csv('sample_submission.csv')

sub_df['InChI'] = [f"InChI=1S/{text}" for text in predictions]
sub_df[['image_id', 'InChI']].to_csv('submission.csv', index=False)



