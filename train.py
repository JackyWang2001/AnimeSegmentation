import torch
from torch import nn
from torch import optim
from model import *


class Experiment:
	def __init__(self, num_classes, loader, device):
		"""
		initialize model, optimizer,
		:param loader: trainloader
		"""
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.loader = loader
		self.model = init_deeplab(150).to(device)
		self.criterion = FocalLoss(num_classes)
		self.optimizer = optim.SGD(self.model.parameters(), lr=1e-5)
		self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.95)

	def train(self, num_epoch):
		train_losses = []
		for epoch in range(num_epoch):
			train_loss = 0.0
			self.model.train()
			for i, (img, mask) in enumerate(self.loader):
				img, mask = img.to(self.device), mask.to(self.device, dtype=torch.long)
				self.optimizer.zero_grad()
				outputs = self.model(img)["out"]
				loss = self.criterion(outputs, mask)
				loss.backward()
				self.optimizer.step()
				train_loss += loss.item()
			train_losses.append(train_loss)
			print("epoch {}".format(epoch), train_loss)
