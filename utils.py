from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.init as init
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn import (Conv2d, CrossEntropyLoss, Linear, MaxPool2d, ReLU,
                      Sequential)
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from mask_detection_dataset import MaskDetectionDataset

def prepare_data(mask_df_path) -> None:
        mask_df = pd.read_pickle(mask_df_path)
        train, validate = train_test_split(mask_df, test_size=0.3, random_state=0,
                                           stratify=mask_df['mask'])
        mask_num = mask_df[mask_df['mask']==1].shape[0]
        non_mask_num = mask_df[mask_df['mask']==0].shape[0]
        not_a_person_num = mask_df[mask_df['mask']==2].shape[0]
        n_samples = [non_mask_num, mask_num]
        normed_weights = [1 - (x / sum(n_samples)) for x in n_samples]
        return [
            MaskDetectionDataset(train),
            MaskDetectionDataset(validate),
            CrossEntropyLoss(weight=torch.tensor(normed_weights))
            ]

def train_dataloader(train_df) -> DataLoader:
    return DataLoader(train_df, batch_size=32, shuffle=True, num_workers=4)

def val_dataloader(validate_df) -> DataLoader:
    return DataLoader(validate_df, batch_size=32, num_workers=4)

train_df, validate_df, cross_entropy_loss = prepare_data("data/dataset/dataset.pickle")
