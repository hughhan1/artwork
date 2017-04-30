import os, sys
import numpy as np
from PIL import Image
import h5py
import glob
import sklearn
import csv

from itertools import chain



def process_images(base_dir, size):
	"""
	Iterate through images in directory, resize to desired dimensions,
	prepare design matrices.

	Args:
		base_dir	: the base directory of images
		size	 	: the desired size of output images

	Returns:
		X_color 	: design matrix of colored images
		X_gray		: design matrix for grayscale
	"""

	#dimensions for design matrices
	N = len(glob.glob1(base_dir, '*.jpg'))
	d = (size[0] * size[1])

	#design matrices, color stacks RBG
	X_color = np.zeros((N, d*3))
	X_gray = np.zeros((N, d))

	idx = 0
	for filename in sorted(os.listdir(base_dir)):

		if filename.endswith('.jpg'):
			sys.stderr.write("Processing: %s\n" % filename)

			#load image and resize
			im = Image.open(os.path.join(base_dir, filename))
			im = im.resize(size, Image.ANTIALIAS)

			#reshape to 1-d vector (and convert to grayscale)
			color_array = np.array(im).ravel()
			gray_array = np.array(im.convert('L')).ravel()

			#load 1-d vector into respective position
			X_color[idx, :] = color_array.T
			X_gray[idx, :] = gray_array.T

			idx += 1

	return X_color, X_gray

def read_labels(filenames):

	img_labels = {}
	for filename in filenames:
		for key, val in csv.reader(open(filename + ".csv")):
			img_labels[key] = val

	
	unique_labels = list(set(img_labels.values()))
	
	class_labels = {} #map of class label to assocaited integer value
	for i in range(len(unique_labels)):
		class_labels[unique_labels[i]] = i


	#write class labels for future use
	w = csv.writer(open("class_labels.csv", "w"))
	for key, val in class_labels.items():
		w.writerow([key, val])

	#design matrices, color stacks RBG
	N = len(img_labels.keys())
	y = np.zeros((N))

	i = 0
	images = sorted(img_labels.keys())
	for img in images:
		label = img_labels[img]
		y[i] = class_labels[label]
		i += 1

	return y
	

def write_dataset(X_color, X_gray, y_nation):
	"""
	Write out design matrices to h5py format.
	To read:
		h5f = h5py.File('artwork.h5','r')
		X_color = h5f['color'][:]
		X_gray = h5f['gray'][:]
		h5f.close()

	Args:
		base_dir	: the base directory of images
		size	 	: the desired size of output images

	Returns:
		X_color 	: design matrix of colored images
		X_gray		: design matrix for grayscale
	"""

	#save to h5py file
	h5f = h5py.File('artwork.h5', 'w')
	h5f.create_dataset('color', data=X_color)
	h5f.create_dataset('gray', data=X_gray)
	h5f.create_dataset('labels', data=y_nation)
	h5f.close()



def main():

	if len(sys.argv) < 2:
		print('error: requires at least one dataset')
		sys.exit(0)


	#resized image dimensions
	#TODO: square okay?
	size = 128, 128

	#base directory
	base_dir = "images/"
	filenames = []
	for i in range(1, len(sys.argv)):
		filenames.append(sys.argv[i])

	X_color, X_gray = process_images(base_dir, size)
	y_nation = read_labels(filenames)
	write_dataset(X_color, X_gray, y_nation)


if __name__ == '__main__':
	main()