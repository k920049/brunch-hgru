import torch
import pandas as pd
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader

from pipeline.torch.Vectorizer import Vectorizer

class BrunchDataset(Dataset):

    def __init__(self,
                 train_df_path="./data/brunch/train.parquet",
                 eval_df_path="./data/brunch/eval.parquet",
                 history_length=128):

        self.history_length = history_length

        self.train_df = pd.read_parquet(train_df_path)
        self.train_size = len(self.train_df)

        self.val_df = pd.read_parquet(eval_df_path)
        self.validation_size = len(self.val_df)

        self._lookup_dict = {
            'train': (self.train_df, self.train_size),
            'valid': (self.val_df, self.validation_size)
        }
        self.set_split('train')

    def set_split(self, split="train"):

        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):

        row = self._target_df.iloc[index]

        length          = row.length
        session_input   = np.pad(row.session_input, (0, self.history_length - length))
        session_output  = np.pad(row.session_output, (0, self.history_length - length))
        session_mask    = np.pad(row.session_mask, (0, self.history_length - length))
        user_mask       = np.pad(row.user_mask, (0, self.history_length - length))
        mask            = np.pad(np.array([1.0] * length), (0, self.history_length - length))

        return {
            'session_input': session_input,
            'session_output': session_output,
            'session_mask': session_mask,
            'user_mask': user_mask,
            'mask': mask
        }

    def get_num_batches(self, batch_size):
        return len(self) // batch_size

def generate_batches(dataset,
                     batch_size,
                     shuffle=True,
                     drop_last=True,
                     device="cpu"):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict