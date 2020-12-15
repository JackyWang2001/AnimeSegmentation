from dataset import *
from model import *
from torch.utils.data import DataLoader

ROOT = os.path.abspath("/media/jpl/T7/ADE20K")

train_dataset = ADE20K_Dataset(ROOT)
train_loader = DataLoader(train_dataset, batch_size=64, num_workers=48)


print(model)