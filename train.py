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
		self.model = init_deeplab(num_classes).to(device)
		self.criterion = nn.CrossEntropyLoss(ignore_index=255)
		self.optimizer = optim.SGD(self.model.parameters(), lr=1e-5)
		self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.95)

	def train(self, num_epoch):
		train_losses = []
		for epoch in range(num_epoch):
			train_loss = 0.0
			self.model.train()
			for i, (img, mask) in enumerate(self.loader):
				img, mask = img.to(self.device), mask.to(self.device)
				self.optimizer.zero_grad()
				outputs = self.model(img)["out"]
				loss = self.criterion(outputs, mask)
				train_loss += loss.item()
				loss.backward()
				self.optimizer.step()
			train_losses.append(train_loss)
			print("epoch {}: ".format(epoch), train_loss)