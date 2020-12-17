import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


def init_deeplab(num_classes=21):
	return torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)


