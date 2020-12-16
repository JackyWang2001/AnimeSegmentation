from torch.utils.data import DataLoader
from dataset import *
from model import *
from train import Experiment

ROOT = os.path.abspath("/media/jpl/T7/ADE20K")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = ADE20K_Dataset(ROOT)
train_loader = DataLoader(train_dataset, batch_size=32, num_workers=48)

exp = Experiment(150, train_loader, device)
exp.train(10)