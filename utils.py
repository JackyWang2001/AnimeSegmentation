import os
import glob


def get_subfolder(path):
	"""
	get all subdirectories
	:param path: string path
	:return: a list of all sub-folders in dir
	"""
	return sorted(glob.glob(os.path.join(path, "*")))