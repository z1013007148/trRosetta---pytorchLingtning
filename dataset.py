import pytorch_lightning as pl
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn.functional as F


def d():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class trRosettaDataset(Dataset):
    def __init__(self, path, mode):
        self.path = path
        self.mode = mode
        self.files = os.listdir(self.path)
        train_len = int(len(self.files) * 0.95)
        test_len = int(len(self.files) * 0.)
        val_len = len(self.files) - train_len - test_len
        self.train_set, self.rest = random_split(self.files, [train_len, test_len + val_len])
        self.test_set, self.val_set = random_split(self.rest, [test_len, val_len])

    def __len__(self):
        if self.mode == "train":
            return len(self.train_set)
        elif self.mode == "test":
            return len(self.test_set)
        else:
            return len(self.val_set)

    def __getitem__(self, item):
        if self.mode == "train":
            path = os.path.join(self.path, self.train_set[item])
        elif self.mode == "test":
            path = os.path.join(self.path, self.test_set[item])
        else:
            path = os.path.join(self.path, self.val_set[item])
        file = np.load(path)
        try:
            if file['x_feature'].shape[-1] < 1:
                print("error")
        except Exception as e :
            print(path)
        return file


def collate_fn(batch_dic):
    batch_len = len(batch_dic)

    max_tensor_length = max(batch_dic[i]['x_feature'].shape[-1] for i in range(batch_len))

    new_batch = [torch.tensor([]).to(d()) for _ in range(5)]
    for i in range(batch_len):
        pad_length = max_tensor_length - batch_dic[i]['x_feature'].shape[-1]
        tensor = [torch.from_numpy(batch_dic[i]['x_feature']).to(d()),
                  torch.from_numpy(batch_dic[i]['y_lable_data_phi6d']).to(d()),
                  torch.from_numpy(batch_dic[i]['y_lable_data_theta6d']).to(d()),
                  torch.from_numpy(batch_dic[i]['y_lable_data_omega']).to(d()),
                  torch.from_numpy(batch_dic[i]['y_lable_data_dist']).to(d())]
        if pad_length > 0:
            for j in range(5):
                new_batch[j] = torch.cat(
                    (new_batch[j], F.pad(tensor[j], (0, pad_length, 0, pad_length), "constant", 0)))
        else:
            for j in range(5):
                new_batch[j] = torch.cat((new_batch[j], tensor[j]))

    return new_batch


class trRosettaDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=6, data_dir=''):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir

    def setup(self, stage): #
        # train/val split
        self.train_dataset = trRosettaDataset(self.data_dir, "train")
        self.val_dataset = trRosettaDataset(self.data_dir, "val")
        self.test_dataset = trRosettaDataset(self.data_dir, "test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=collate_fn) # num_workers=2,pin_memory=True

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=collate_fn)
