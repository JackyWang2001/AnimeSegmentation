from utils import *

ROOT = os.path.abspath("/media/jpl/T7/ADE20K")
TRAINING_PATH = os.path.join(ROOT, "images", "training")
training_dirs = get_subfolder(TRAINING_PATH)
training_dirs = concat_subfolder(training_dirs)
training_raw = get_img_path(training_dirs)

VALIDATION_PATH = os.path.join(ROOT, "images", "validation")
validation_dirs = get_subfolder(VALIDATION_PATH)
validation_dirs = concat_subfolder(validation_dirs)
validation_raw = get_img_path(validation_dirs)

copy_to_new(training_raw, "/media/jpl/T7/_Documents/CartoonGAN/ADE/training")
copy_to_new(validation_raw, "/media/jpl/T7/_Documents/CartoonGAN/ADE/validation")