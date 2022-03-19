import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from skimage.color import rgb2gray
from torch.utils import data
from torchvision.transforms import ToTensor
####################################  Load Data #####################################



# ===== normalize over the dataset
def dataset_normalized(imgs):
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs - imgs_mean) / imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (
                    np.max(imgs_normalized[i]) - np.min(imgs_normalized[i]))) * 255
    return imgs_normalized




class Skin_loader(data.Dataset):
    def __init__(self, imgs_train, masks_train):

        self.imgs_train = imgs_train[0:int(len(imgs_train))]
        self.masks_train = masks_train[0:int(len( masks_train))]

    def __len__(self):
        return len(self.imgs_train)

    def __getitem__(self, index):
        image = self.imgs_train[index]
        label = self.masks_train[index]

        return ToTensor()(image).type(torch.FloatTensor), ToTensor()(label).type(torch.FloatTensor)
