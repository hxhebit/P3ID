from torchvision import datasets, transforms
import torch
from torch.utils.data.distributed import DistributedSampler  
import numpy as np


def load_data(data_folder, batch_size, train, mean, std, args, num_workers=0, **kwargs):

    transform = {
        'train': transforms.Compose(
            [transforms.Resize([256, 256]),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),
            transforms.Normalize(mean=mean, 
                                  std=std)]), 
        'test': transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,
                                  std=std)])
    }
    data = datasets.ImageFolder(root=data_folder, transform=transform['train' if train else 'test'])

    train_sampler = DistributedSampler(data) if train else None

    data_loader = get_data_loader(data, train_sampler, batch_size=batch_size, 
                                shuffle=False, 
                                num_workers=num_workers, **kwargs, drop_last=True if train else False)
    n_class = len(data.classes)
    return data_loader, n_class


def get_data_loader(dataset, train_sampler, batch_size, shuffle, drop_last=False, num_workers=0, infinite_data_loader=False, **kwargs):
    if not infinite_data_loader:
        return torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers,  **kwargs)
    else:
        return InfiniteDataLoader(dataset, sampler=train_sampler, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, **kwargs)

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, num_workers=0, weights=None, **kwargs):
        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=False,
                num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=False)
            
        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=drop_last)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        return 0 # Always return 0