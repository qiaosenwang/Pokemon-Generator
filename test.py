from random import shuffle
import torch, torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import POKEMON
import os

DATA_DIR = '/home/wangqs/Data/POKEMON128/'
FIGS_DIR = './figs/'
batch_size = 8

if not os.path.exists(FIGS_DIR):
        os.makedirs(FIGS_DIR)


transform = transforms.ToTensor()

train_dataset = POKEMON(root_dir=DATA_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)



for i, images in enumerate(train_loader):
    if i == 0:
        print(images.size())
        images = images.view(images.size(0), 3, 128, 128)
        save_image(images, os.path.join(FIGS_DIR, 'images-{}.png'.format(i+1)))


