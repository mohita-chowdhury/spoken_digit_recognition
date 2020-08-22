import torch
import torch.nn as nn
import os
import numpy as np


device = torch.device("cuda")


class Flatten(nn.Module):
   def __init__(self):
        super(Flatten,self).__init__()
 
   def forward(self, x):
        # return x.view(x.size()[0], -1)
        return torch.flatten(x)

class SpeechConv(nn.Module):
	def __init__(self):
		super(SpeechConv, self).__init__()

		self.flat = Flatten()

		self.ConvNet = nn.Sequential(
						nn.Conv2d(3, 16, kernel_size = (7,7), stride = (1,1), bias = False),
						nn.ReLU(inplace = False),
						nn.MaxPool2d(kernel_size=(3, 3), stride = (2,2)),
						# nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

						nn.Conv2d(16, 32, kernel_size = (5,5), stride = (1,1), bias = False),
						nn.ReLU(inplace = False),
						nn.MaxPool2d(kernel_size=(3, 3), stride = (2,2)),
						# nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

						nn.Conv2d(32, 64, kernel_size = (3,3), stride = (1,1), bias = False),
						nn.ReLU(inplace = False),
						nn.MaxPool2d(kernel_size=(3, 3), stride = (2,2)),
						# nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

						nn.Conv2d(64, 128, kernel_size = (3,3), stride = (1,1), bias = False),
						nn.ReLU(inplace = False),
						nn.MaxPool2d(kernel_size=(3, 3), stride = (2,2)),
						# nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

						nn.Conv2d(128, 128, kernel_size = (3,3), stride = (1,1), bias = False),
						nn.ReLU(inplace = False),
						nn.MaxPool2d(kernel_size=(3, 3), stride = (2,2)),

						nn.Conv2d(128, 64, kernel_size = (1,1), stride = (1,1), bias = False),
						nn.ReLU(inplace = False)
						# nn.MaxPool2d(kernel_size=(3, 3), stride = (2,2))
						# nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

						# nn.Conv2d(128, 256, kernel_size = (3,3), stride = (1,1), bias = False),
						# nn.ReLU(inplace = False),
						# nn.MaxPool2d(kernel_size=(3, 3), stride = (2,2))
						# # nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
						)
		self.Project = nn.Sequential(
						nn.Linear(1024, 64),
						nn.ReLU(),
						# nn.BatchNorm1d(1024),
						nn.Dropout(),
						nn.Linear(64, 10)
						)

	def forward(self, spec):
		

		torch.transpose(spec, 2, 3)
		spec = spec.permute(0,3,1,2)
		spec_processed = self.ConvNet(spec.float())
		# import pdb; pdb.set_trace()

		spec_flattened = self.flat(spec_processed)
		spec_projected = self.Project(spec_flattened)

		return spec_projected



