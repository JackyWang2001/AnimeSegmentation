import torch
from torch import nn
from PIL import Image


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

# 33882
# tensor([0.5169, 0.4734, 0.4078])
# tensor([0.2075, 0.2059, 0.1907])