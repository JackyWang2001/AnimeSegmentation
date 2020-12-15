from utils import *

ROOT = os.path.abspath("/media/jpl/T7/ADE20K")
TRAINING_PATH = os.path.join(ROOT, "images", "training")
VALIDATION_PATH = os.path.join(ROOT, "images", "validation")
CARTOON_TRAIN_PATH = os.path.abspath("/media/jpl/T7/_Documents/CartoonGAN/ADE/training")
CARTOON_VAL_PATH = os.path.abspath("/media/jpl/T7/_Documents/CartoonGAN/ADE/validation")
OUTPUT_TRAIN_PATH = os.path.abspath("/media/jpl/T7/_Documents/CartoonGAN/Training_output")
OUTPUT_VAL_PATH = os.path.abspath("/media/jpl/T7/_Documents/CartoonGAN/Validation_output")

training_dirs = get_subfolder(TRAINING_PATH)
training_dirs = concat_subfolder(training_dirs)
training_raw = get_img_path(training_dirs)

validation_dirs = get_subfolder(VALIDATION_PATH)
validation_dirs = concat_subfolder(validation_dirs)
validation_raw = get_img_path(validation_dirs)

# comment below if have already run this to get copy of images
copy_to_new(training_raw, CARTOON_TRAIN_PATH)
copy_to_new(validation_raw, CARTOON_VAL_PATH)

# run following from terminal in the dir of CartoonGAN, which is an improved version of CycleGAN
# we use implementation from https://github.com/Yijunmaverick/CartoonGAN-Test-Pytorch-Torch
# this implementation provides 4 pretrained models: Hayao, Hosoda, Paprika, Shinkai
# python test.py --input_dir /media/jpl/T7/_Documents/CartoonGAN/ADE/training --style Hayao --gpu 1
# python test.py --input_dir /media/jpl/T7/_Documents/CartoonGAN/ADE/training --style Hosoda --gpu 1
# python test.py --input_dir /media/jpl/T7/_Documents/CartoonGAN/ADE/validation --style Hayao --gpu 1
# python test.py --input_dir /media/jpl/T7/_Documents/CartoonGAN/ADE/validation --style Shinkai --gpu 1

# comment below if have already put back transferred images to dataset dir
put_back(training_raw, OUTPUT_TRAIN_PATH)
put_back(validation_raw, OUTPUT_VAL_PATH)