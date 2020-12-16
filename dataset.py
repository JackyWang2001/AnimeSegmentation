import torch
import os
import glob
import numpy as np
from torch import nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils import *


class ADE20K_Dataset(Dataset):
    """
    ADE20K dataset with anime style transfer
    """
    def __init__(self, root, transform=None, status="Train"):
        super(ADE20K_Dataset, self).__init__()
        self.root = root
        self.transform = transform
        if status == "Train":
            self.dir = os.path.join(self.root, "images", "training")
            self.styles = ("Hayao", "Hosoda")
        else:
            self.dir = os.path.join(self.root, "images", "validation")
            self.styles = ("Hayao", "Shinkai")
        self.classes_path = concat_subfolder(get_subfolder(self.dir))
        # save mask as a dict because one mask corresponds to two images
        self.images_path, self.masks_path = [], {}
        for folder in self.classes_path:
            for style in self.styles:
                style = "*_" + style + ".jpg"
                for img in glob.glob(os.path.join(folder, style)):
                    self.images_path.append(img)
            for mask in glob.glob(os.path.join(folder, "*_seg.png")):
                img_name = mask.split("/")[-1].replace("_seg.png", "")
                self.masks_path[img_name] = mask

    def __getitem__(self, index):
        """
        get image and mask,
        :param index:
        :return:
        """
        img_path = self.images_path[index]
        img_name = img_path.split("/")[-1]
        for style in self.styles:
            style = "_" + style + ".jpg"
            img_name = img_name.replace(style, "")
        mask_path = self.masks_path[img_name]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        # resize mask into the same dim with CartoonGAN outputs
        mask = mask.resize((224, 224), resample=Image.NEAREST)
        # convert mask img into labels
        temp = Image.new("L", (224, 224), 0)
        temp.paste(mask, (0, 0))
        mask = torch.from_numpy(np.array(temp))
        # apply transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # compute normalizing terms in prepare_dataset.py
                transforms.Normalize(mean=[0.5169, 0.4734, 0.4078], std=[0.2075, 0.2059, 0.1907])
            ])
        img = self.transform(img)
        return img, mask

    def __len__(self):
        return len(self.images_path)

