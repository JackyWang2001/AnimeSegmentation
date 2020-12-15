import torch
import torchvision


model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)