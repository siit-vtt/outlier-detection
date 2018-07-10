import torch.nn as nn
import torch.nn.functional as F
import torch


class _net(nn.Module):

	def __init__(self, n_classes, nc):
		super(_net, self).__init__()

		self.n_classes = n_classes

		self.conv_block = nn.Sequential(
			nn.Conv2d(nc, 64, 3, padding=1), 
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2),
			nn.Conv2d(64, 128, 3, padding=1), 
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2),
			nn.Conv2d(128, 128, 3, padding=1), 
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2),
			)

		self.classifier = nn.Linear(2048, n_classes)

	def forward(self, x):

		x = self.conv_block(x)

		x = x.view(x.shape[0], -1)
		x = self.classifier(x)

		return x