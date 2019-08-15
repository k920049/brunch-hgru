import pandas as pd
import numpy as np
import os

from fire import Fire
from pipeline.Reader import Reader

class Datasets(Reader):

    def __init__(self, data_path):

        self.data_path = data_path

        super(Datasets, self).__init__()

    def read(self):

        train_file  = os.path.join(self.data_path, "train.npy")
        label_file  = os.path.join(self.data_path, "label.npy")
        mask_file   = os.path.join(self.data_path, "mask.npy")

        candidate_train = np.transpose(np.load(train_file), axes=[0, 2, 1])
        candidate_label = np.transpose(np.load(label_file), axes=[0, 2, 1])
        candidate_mask = np.transpose(np.load(mask_file), axes=[0, 2, 1])

        message = "At least one of the dimension in the input data doesn't match with others"
        assert len(candidate_train) == len(candidate_label), message
        assert len(candidate_label) == len(candidate_mask), message
        assert len(candidate_mask) == len(candidate_train), message

        train = []
        label = []
        mask = []
        for i in range(len(candidate_train)):
            if np.sum(candidate_mask[i]) >= 10.0:
                train.append(candidate_train[i])
                label.append(candidate_label[i])
                mask.append(candidate_mask[i])

        return np.array(train)[0:5000], np.array(label)[0:5000], np.array(mask)[0:5000]

    def write(self):
        pass

    def show(self, n):
        train, label, mask = self.read()

        print(train.shape, label.shape, mask.shape)

        # train = train[:n]
        # label = label[:n]
        # mask = mask[:n]
        #
        # for t, l, m in zip(train, label, mask):
        #     for elem_t, elem_l, elem_m in zip(t, l, m):
        #         print(elem_t, elem_l, elem_m)


if __name__ == "__main__":
    Fire(Datasets)