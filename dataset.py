import torch
from torch import nn


class ADE20K_Dataset(nn.module):
	"""
	ADE20K dataset with anime style transfer
	"""
	def __init__(self):
		super(ADE20K_Dataset, self).__init__()

	def __getitem__(self, item):
		return

	def __len__(self):
		return