import pandas as pd
import numpy as np
import os

from pipeline.Reader import Reader

class Datasets(Reader):

    def __init__(self, data_path):

        self.data_path = data_path

        super(Datasets, self).__init__()

    def read(self):

        train_file  = os.path.join(self.data_path, "train.npy")
        label_file  = os.path.join(self.data_path, "label.npy")
        mask_file   = os.path.join(self.data_path, "mask.npy")

        return np.load(train_file), np.load(label_file), np.load(mask_file)

    def write(self):
        pass