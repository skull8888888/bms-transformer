import torch

from tokenizer import Tokenizer

tokenizer = torch.load('./tokenizer2.pth')

from model import Model
from dataset import TrainDataset, get_transforms, bms_collate

from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset

import pandas as pd
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from config import CFG
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold




train = pd.read_csv('train_tokenized.csv')
folds = train.copy()
Fold = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds['InChI_length'])):
    folds.loc[val_index, 'fold'] = int(n)

fold = 0
trn_idx = folds[folds['fold'] == fold].index
val_idx = folds[folds['fold'] == fold + 1].index[:10000]

train_folds = folds.loc[trn_idx].reset_index(drop=True)
valid_folds = folds.loc[val_idx].reset_index(drop=True)

train_dataset = TrainDataset(train_folds, tokenizer, transform=get_transforms(data='train'))
valid_dataset = TrainDataset(valid_folds, tokenizer, transform=get_transforms(data='valid'))

train_loader = DataLoader(train_dataset, 
                          batch_size=CFG.batch_size, 
                          shuffle=True, 
                          num_workers=CFG.num_workers, 
                          pin_memory=True,
                          drop_last=True, 
                          collate_fn=bms_collate)
valid_loader = DataLoader(valid_dataset, 
                          batch_size=CFG.batch_size, 
                          shuffle=False, 
                          num_workers=CFG.num_workers, 
                          pin_memory=True,
                          drop_last=True, 
                          collate_fn=bms_collate,
                             )

model = Model(tokenizer, len(train_loader))
val_acc_callback = ModelCheckpoint(
        monitor='LD', 
#         dirpath=checkpoint_dir,
        filename='{epoch:02d}-{LD:.4f}',
        save_last=True, 
        mode='min')

trainer = Trainer(
    gpus=[0], 
    accelerator='ddp',
    callbacks=[val_acc_callback], 
    precision=16,
    max_epochs=CFG.epochs)

trainer.fit(model, train_loader, valid_loader)