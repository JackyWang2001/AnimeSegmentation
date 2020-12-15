from utils import *

ROOT = os.path.abspath("/media/jpl/T7/ADE20K")
TRAINING_PATH = os.path.join(ROOT, "images", "training")
VALIDATION_PATH = os.path.join(ROOT, "images", "validation")
TRAINING_DIRS = get_subfolder(TRAINING_PATH)
VALIDATION_DIRS = get_subfolder(VALIDATION_PATH)

