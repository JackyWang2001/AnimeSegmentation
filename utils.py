import os
import glob
import shutil


def get_subfolder(path):
	"""
	get all subdirectories
	:param path: string path
	:return: a list of all sub-folders in dir
	"""
	return sorted(glob.glob(os.path.join(path, "*")))


def concat_subfolder(dir_list):
	"""
	concat all subdirectories
	:param dir_list: list of dirs like [training/a, training/b, ...]
	:return: a list of path point to paths like [training/a/abbey, training/a/airfield, ...]
	"""
	results = []
	for path in dir_list:
		results += get_subfolder(path)
	return results


def get_img_path(dir_list):
	"""
	get all images for style transferring, raw images are .jpg files
	:param dir_list: a list of path point to paths like [training/a/abbey, training/a/airfield, ...]
	:return: a dict of paths of raw images, key is image name and value is original directory and path
	"""
	raw_images = {}
	for path in dir_list:
		for img in glob.glob(os.path.join(path, "*.jpg")):
			img_name = img.split("/")[-1]
			raw_images[img_name] = [path, img]
	return raw_images


def copy_to_new(img_dict, destination):
	"""
	copy raw images to destination
	"""
	for img_name in img_dict:
		shutil.copy(img_dict[img_name][1], destination)
		print("copying " + img_name)
