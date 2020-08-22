import torch
import torch.nn as nn
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim 
import sys
import warnings
import random
warnings.filterwarnings("ignore") 

from torch.utils.data import DataLoader
from torchvision import transforms, utils

import model
from train import train
from test import test
from dataloader import speechDataset


PATH = '/scratch/shared/nfs1/mohita/ufonia/speech_rec_pytorch/saved/models/speech_net_aug.pth.tar'
device = torch.device("cuda")

seed=10
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
def _init_fn(worker_id):
    np.random.seed(int(seed))

parser=argparse.ArgumentParser()
parser.add_argument('-e','--number_of_epochs',type=int,required=False,default=10,help='Number of epochs to run the model')
parser.add_argument('-l','--learning_rate',type=float,required=False,default=0.01,help='Initial learning_rate')
parser.add_argument('-sa','--saving_epoch',type=int,required=False,default=1000,help='In which epoch to save e.g. 10 means the model would be saved every tenth epoch while training')
parser.add_argument('-fn','--folder_name',type=str,required=False,default='result',help='Name of the folder in which you want to save')
parser.add_argument('-tl','--tb_path',type=str,required=False,default='result',help='Name of the folder in which you want to save tensorboard logs')
parser.add_argument('-ba','--batch_size',type=int,required=False,default=1,help='Batch size')
parser.add_argument('-p','--pretrained',type=str,required=False,default='f',help='If this flag is t then a pretrained model will be loaded')
parser.add_argument('-c','--Check_point',type=str,required=False,default='best',help='First word of the checkpoint file which would be followed by checkpoint.pth.tar')

args=parser.parse_args()

number_of_epochs=args.number_of_epochs
learning_rate=args.learning_rate
saving_epoch=args.saving_epoch
folder_name=args.folder_name
tb_path=args.tb_path
batch_size=args.batch_size
pretrained=args.pretrained
check=args.Check_point

params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 6}

with open('data/partition_aug.json') as json_file:
    partition = json.load(json_file)

with open('data/labels_aug.json') as json_file:
    labels = json.load(json_file)

train_IDs = partition['train']
test_IDs = partition['test']

training_set = speechDataset(partition['train'], labels, 'train')
training_generator = torch.utils.data.DataLoader(training_set, **params)

test_set = speechDataset(partition['test'], labels, 'test')
test_generator = torch.utils.data.DataLoader(test_set, **params)


speech_model = model.SpeechConv()
speech_model.to(device)
optimizer = optim.Adam(speech_model.parameters(), lr=0.0001)

# train(training_generator, number_of_epochs, batch_size, speech_model, optimizer)
speech_model.load_state_dict(torch.load(PATH))
test(test_generator, speech_model)

