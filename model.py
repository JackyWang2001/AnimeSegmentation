import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


def init_deeplab(num_classes=21):
	return torchvision.models.segmentation.deeplabv3_resnet50(num_classes=num_classes)


class FocalLoss(nn.Module):
	def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
		super(FocalLoss, self).__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.reduction = reduction
		self.criterion = nn.BCEWithLogitsLoss(reduction='none')

	def forward(self, logits, label):
		"""
        logits and label have same shape, and label data type is long
        args:
            logits: tensor of shape (N, ...)
            label: tensor of shape(N, ...)
        """
		# compute loss
		logits = logits.float()  # use fp32 if logits is fp16
		with torch.no_grad():
			alpha = torch.empty_like(logits).fill_(1 - self.alpha)
			alpha[label == 1] = self.alpha

		probs = torch.sigmoid(logits)
		pt = torch.where(label == 1, probs, 1 - probs)
		ce_loss = self.criterion(logits, label.float())
		loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
		if self.reduction == 'mean':
			loss = loss.mean()
		if self.reduction == 'sum':
			loss = loss.sum()
		return loss


