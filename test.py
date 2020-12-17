import torch
from PIL import Image
from torchvision import transforms
from model import *


class Predictor:
	def __init__(self, img_path, model_file="model.pt"):
		self.img_path = img_path
		self.model_file = model_file
		self.model = self._get_model()

	def _get_model(self):
		model = init_deeplab()
		model.load_state_dict(torch.load(self.model_file))
		return model

	def pred(self):
		image = Image.open(self.img_path)
		transform = transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			# compute normalizing terms in prepare_dataset.py
			transforms.Normalize(mean=[0.5169, 0.4734, 0.4078], std=[0.2075, 0.2059, 0.1907])
		])
		image = transform(image)
		outputs = self.model(image)