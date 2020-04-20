"""Module allows to access images from GTA Crimes dataset"""

import tarfile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor


class GTADataset(Dataset):
    """Provides access to GTA Crime images dataset"""

    CLASSES = {
        'Normal': 0,
        'Arrest': 1,
        'Arson': 2,
        'Assault': 3,
        'Explosion': 4,
        'Fight': 5,
        'Robbery': 6,
        'Shooting': 7,
        'Vandalism': 8,
    }

    def __init__(self, archive_file: str, transform=None):
        self.archive = tarfile.open(archive_file, 'r:gz')
        self.transform = transform
        self.closed = False

    def __len__(self):
        if self.closed:
            raise Exception('Dataset has been closed')
        return len(self.archive.getnames())

    def __getitem__(self, idx):
        if self.closed:
            raise Exception('Dataset has been closed')

        if torch.is_tensor(idx):
            idx = idx.tolist()

        name = self.archive.getnames()[idx]
        image_class = name.split('/')[0]
        image_file = self.archive.extractfile(name)
        image_data = Image.open(image_file)

        if self.transform:
            image_data = self.transform(image_data)

        return image_data, self.CLASSES[image_class]

    def close(self):
        """Close the underlying archive file"""
        if not self.closed:
            self.archive.close()
            self.closed = True


def create_gta_dataloaders(dataset_path, transform=ToTensor(), batch_size=16):
    dataset = ImageFolder(dataset_path, transform=transform)
    shuffled_indices = np.random.permutation(len(dataset))
    train_idx = shuffled_indices[:int(0.8 * len(dataset))]
    val_idx = shuffled_indices[int(0.8 * len(dataset)):]

    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              drop_last=True,
                              sampler=SubsetRandomSampler(train_idx),
                              num_workers=1,
                              pin_memory=True)
    val_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            drop_last=False,
                            sampler=SubsetRandomSampler(val_idx),
                            num_workers=1,
                            pin_memory=True)
    return train_loader, val_loader
