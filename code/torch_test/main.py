#!/bin/env python3

from data import FERDataset
from model import SEResNet

import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, Subset
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import FashionMNIST, FER2013
from torchvision.transforms import v2

learning_rate = 1e-4
batch_size = 64
epochs = 50

# Create model
class ResNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.preprocess = weights.transforms()
        self.feature = resnet18(weights=weights)
        self.feature.requires_grad_(False)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 10),
        )

    def forward(self, x):
        x = self.preprocess(x)
        x = self.feature(x)
        x = self.head(x)
        return x

# Training and testing loops

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    total = 0
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss, current = loss.item(), batch * batch_size + len(X)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        total += len(y)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}], " +
                f'acc: {100*correct/total:>0.1f}%', end='\r',
                flush=True)

    print()


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct, total = 0, 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total += len(y)

    test_loss /= num_batches
    correct /= total
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n",
            flush=True)


if __name__ == '__main__':
    dataset = 'FashionMNIST'

    if dataset == 'FashionMNIST':
        input_size = (3, 28, 28)
        num_classes = 10
    else:
        input_size = (3, 48, 48)
        num_classes = 8

    # model = ResNetClassifier()
    model = SEResNet(input_size=input_size, num_classes=num_classes)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Load data
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if dataset == 'PartialFER2013':
        train_data = FERDataset('./FERPlus/data/FER2013Train')
        test_data = FERDataset('./FERPlus/data/FER2013Valid')
    elif dataset == 'FER2013':
        all_data = FER2013('./data/', 'train',
                transform=transforms)
        print(len(all_data))
        train_data = Subset(all_data, range(0, 20000))
        test_data = Subset(all_data, range(20000, 28709))
    elif dataset == 'FashionMNIST':
        train_data = FashionMNIST('./data/',
                train=True,
                download=True,
                transform=transforms)
        test_data = FashionMNIST('./data/',
                train=False,
                download=True,
                transform=transforms)
    else:
        raise RuntimeError(f'Unrecognized dataset: {dataset}')

    train_sampler = RandomSampler(train_data, replacement=True, num_samples=3000)
    train_dataloader = DataLoader(train_data, batch_size=batch_size,
            sampler=train_sampler)
    test_sampler = RandomSampler(test_data, replacement=True, num_samples=500)
    test_dataloader = DataLoader(test_data, batch_size=batch_size,
            sampler=test_sampler)
    
    print('Done loading data')
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------", flush=True)
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!", flush=True)
 

