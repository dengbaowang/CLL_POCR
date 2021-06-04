import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize, Pad, RandomCrop, RandomHorizontalFlip, RandomErasing
from cifar import MY_CIFAR10
from svhn import MY_SVHN

np.random.seed(2)

def cifar10_dataloaders(data_dir):
    print('Data Preparation')    
    cifar10_train_ds = MY_CIFAR10(data_dir, train=True, download=True)
    train_loader = torch.utils.data.DataLoader(
        cifar10_train_ds,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    print('Loading dataset {0} for training -- Num_samples: {1}, num_classes: {2}'.format(datasets.CIFAR10.__name__,len(cifar10_train_ds),10))

    test_transform = Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # Data loader for test dataset
    cifar10_test_ds = datasets.CIFAR10(data_dir, transform=test_transform, train=False, download=True)
    print('Test set -- Num_samples: {0}'.format(len(cifar10_test_ds)))
    test = DataLoader(
        cifar10_test_ds, batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    return train_loader, test


def svhn_dataloaders(data_dir):
    print('Data Preparation')    
    svhn_train_ds = MY_SVHN(data_dir, split='train', download=True)
    train_loader = torch.utils.data.DataLoader(
        svhn_train_ds,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    print('Loading dataset {0} for training -- Num_samples: {1}, num_classes: {2}'.format(datasets.SVHN.__name__,len(svhn_train_ds),10))

    test_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    # Data loader for test dataset
    svhn_test_ds = datasets.SVHN(data_dir, transform=test_transform, split='test', download=True)
    print('Test set -- Num_samples: {0}'.format(len(svhn_test_ds)))
    test = DataLoader(
        svhn_test_ds, batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    return train_loader, test