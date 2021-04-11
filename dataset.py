import numpy as np

import cv2
from PIL import Image

import torch
import torch.nn as nn

from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip, 
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout, 
    IAAAdditiveGaussianNoise, Transpose, Blur, PadIfNeeded, LongestMaxSize
    )
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

from config import CFG

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def add_pad_mask(mask, pad_len):
    
    default_mask = torch.zeros(pad_len, pad_len)

    default_mask[:mask.size(0),:mask.size(1)] = mask
    
    return default_mask

def bms_collate(batch):
    
    pad_token = 192
    
    imgs, labels, label_lengths = [], [], []
    
    for data_point in batch:
        
        imgs.append(data_point[0])
        
        labels.append(data_point[1])
        
        label_lengths.append(data_point[2])
                
    labels = pad_sequence(labels, batch_first=True, padding_value=pad_token)
    
    pad_masks = labels == pad_token
    
    mask = generate_square_subsequent_mask(labels.size(1))
    
    return torch.stack(imgs), labels, torch.stack(label_lengths).reshape(-1, 1), mask, pad_masks


def get_transforms(*, data):
    
    if data == 'train':
        return Compose([
            LongestMaxSize(CFG.size),
            PadIfNeeded(min_height=CFG.size , min_width=CFG.size , border_mode=1),
#             Resize(CFG.size,CFG.size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    
    elif data == 'valid':
        return Compose([
            LongestMaxSize(CFG.size),
            PadIfNeeded(min_height=CFG.size , min_width=CFG.size , border_mode=1),
#             Resize(CFG.size,CFG.size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    
# ====================================================
# Dataset
# ====================================================
class TrainDataset(Dataset):
    def __init__(self, df, tokenizer, transform=None):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.file_paths = df['file_path'].values
        self.labels = df['InChI_text'].values
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = cv2.imread('../' + file_path[2:])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        label = self.labels[idx]
        label = self.tokenizer.text_to_sequence(label)
        label_length = len(label)
        label_length = torch.LongTensor([label_length])
        return image, torch.LongTensor(label), label_length
    


class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.df = df
        self.file_paths = df['file_path'].values
        self.transform = transform
        self.flip = Compose([Transpose(p=1), VerticalFlip(p=1)])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
#         image_id = self.df.iloc[idx]['image_id']
        
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        h, w, _ = image.shape
        if h > w:
            image = self.flip(image=image)['image']

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image