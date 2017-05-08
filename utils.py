import os, sys
import numpy as np
from PIL import Image
import h5py
import glob
import sklearn
import csv

from itertools import chain

class_labels = {}
nation_labels = {}
date_labels = {} #global so can be used elsewhere
processed_img = []

def label_mapping(filename, label_map):
	"""
	Creates mapping of filename to label from CSV

	Args:
		filenames	: the names of datasets to be used (moma, getty, etc)
	"""
	for key, val in csv.reader(open(filename + ".csv")):
		label_map[key + '.jpg'] = val

def clean_imgs(base_dir):
	"""
	Deletes any images from directory that are unusable -
	ie are not RGB, do not have label
	"""

	for filename in sorted(os.listdir(base_dir)):
		if filename.endswith('.jpg'):
			if filename not in class_labels.keys() or filename not in nation_labels.keys() or filename not in date_labels.keys():
				os.rename(os.path.join(base_dir, filename), os.path.join(base_dir+'faulty/', filename))
			else:
				im = Image.open(os.path.join(base_dir, filename))
				if(im.mode != 'RGB'):
					os.rename(os.path.join(base_dir, filename), os.path.join(base_dir+'faulty/', filename))


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

			processed_img.append(filename)

			idx += 1


	print(idx)
	return X_color, X_gray

def read_labels(label_mapping, filename):

	
	unique_labels = list(set(label_mapping.values()))
	
	class_labels = {} #map of class label to assocaited integer value
	for i in range(len(unique_labels)):
		class_labels[unique_labels[i]] = i


	#write class labels for future use
	w = csv.writer(open(filename + ".csv", "w"))
	for key, val in class_labels.items():
		w.writerow([key, val])

	#design matrices, color stacks RBG
	N = len(processed_img)
	y = np.zeros((N))

	i = 0
	images = sorted(processed_img)
	for img in images:
		label = label_mapping[img]
		y[i] = class_labels[label]
		i += 1

	return y
	

def write_dataset(X_color, X_gray, y_class, y_nation, y_date):
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
	h5f.create_dataset('class', data=y_class)
	h5f.create_dataset('nation', data=y_nation)
	h5f.create_dataset('date', data=y_date)
	h5f.close()



def main():

	#if len(sys.argv) < 2:
	#	print('error: requires at least one dataset')
	#	sys.exit(0)


	#resized image dimensions
	#TODO: square okay?
	size = 64, 64

	#base directory
	base_dir = "images/"

	#if using multiple datasets, not used for now
	filenames = []
	for i in range(1, len(sys.argv)):
		filenames.append(sys.argv[i])



	label_mapping('moma_class', class_labels)
	label_mapping('moma_nation', nation_labels)
	label_mapping('moma_date', date_labels)
	clean_imgs(base_dir)
	X_color, X_gray = process_images(base_dir, size)
	y_class = read_labels(class_labels, 'class_labels')
	y_nation = read_labels(nation_labels, 'nation_labels')
	y_date = read_labels(date_labels, 'date_labels')

	print(X_color.shape)
	print(X_gray.shape)
	print(y_class.shape)
	print(y_nation.shape)
	print(y_date.shape)
	write_dataset(X_color, X_gray, y_class, y_nation, y_date)


if __name__ == '__main__':
	main()