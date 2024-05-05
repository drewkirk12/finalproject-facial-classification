import csv
import itertools
import multiprocessing as mp
import numpy as np
import operator
import os
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.io import ImageReadMode
from torchvision.transforms import v2

class FERDataset(Dataset):
    all_emotions = None
    all_images = None
    split = None

    def __init__(self, data_path):
        self.data_path = data_path
        print(f'Loading data from {data_path}')
        self.images, self.sizes, self.emotions = read_files(data_path)
        print('Data loaded', flush=True)

        self.transforms = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_path, self.images[idx])
        image = torchvision.io.read_image(image_path, ImageReadMode.RGB)
        # image = image.float() / 255
        image = self.transforms(image)
        return image, self.emotions[idx]

def read_files(path):
    with mp.Pool(processes=8) as pool:
        with open(os.path.join(path, './label.csv')) as file:
            reader = csv.reader(file)
            images = pool.starmap(operator.getitem, itertools.product(reader, (0,)))

        with open(os.path.join(path, './label.csv')) as file:
            reader = csv.reader(file)
            sizes = pool.starmap(operator.getitem, itertools.product(reader, (1,)))

        with open(os.path.join(path, './label.csv')) as file:
            reader = csv.reader(file)
            emotions = pool.starmap(get_emotion, itertools.product(reader))

    print('Done reading', flush=True)
    images = np.array(images)
    sizes = np.array(sizes)
    emotions = np.array(emotions)

    valid = images != ''
    images = images[valid]
    sizes = sizes[valid]
    emotions = emotions[valid]

    n = 512
    images = images[:n]
    sizes = sizes[:n]
    emotions = emotions[:n]

    return images, sizes, emotions

def get_emotion(line):
    num_emotions = 10
    emotion = line[2:2+num_emotions]
    emotion = np.array([int(e) for e in emotion])
    # emotion = emotion.astype(dtype='float32') / 10
    # assert(np.shape(emotion) == (num_emotions,))
    # Get only the largest
    emotion = np.argmax(emotion)
    return emotion

