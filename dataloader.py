from __future__ import print_function, division
import os
import torch
# import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class speechDataset:

	def __init__(self, list_IDs, labels, phase):
	    """Initialization
	    """
	    self.labels = labels
	    self.list_IDs = list_IDs
	    self.phase = phase

	def __len__(self):
	    """Denotes the total number of samples'
	    """
	    return len(self.list_IDs)

	def __getitem__(self, index):
	    """Generates one sample of data'
	    """
	    ID = self.list_IDs[index]

	    # Load data and get label
	    X = cv2.imread('data/' + self.phase + '/' + ID)
	    y = self.labels[ID]
	    
	    return X, y


