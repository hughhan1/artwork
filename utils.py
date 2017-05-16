import os, sys
import numpy as np
from PIL import Image
import h5py
import glob
import sklearn
import csv

from itertools import chain

artist_labels = {}
style_labels = {}
genre_labels = {}
date_labels = {} #global so can be used elsewhere

def intersect(a, b):
	return list(set(a) & set(b))


def label_mapping(filename, label_map):
	"""
	Creates mapping of filename to label from CSV
	Args:
		filenames	: the names of datasets to be used (moma, getty, etc)
	"""
	for key, val in csv.reader(open(filename)):
		label_map[key] = val

#def clean_imgs(base_dir):
#	"""
#	Deletes any images from directory that are unusable -
#	ie are not RGB, do not have label
#	"""
#
#	for filename in sorted(os.listdir(base_dir)):
#		if filename.endswith('.jpg'):
#			if filename not in class_labels.keys() or filename not in nation_labels.keys() or filename not in date_labels.keys():
#				os.rename(os.path.join(base_dir, filename), os.path.join(base_dir+'faulty/', filename))
#			else:
#				im = Image.open(os.path.join(base_dir, filename))
#				if (im.mode != 'RGB'):
#					os.rename(os.path.join(base_dir, filename), os.path.join(base_dir+'faulty/', filename))


def process_images(base_dir, size, labels):
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

	image_filenames = glob.glob1(base_dir, '*.jpg')
	mapping_filenames = labels.keys()
	dataset = intersect(image_filenames, mapping_filenames)

	N = len(dataset)

	# N = len(glob.glob1(base_dir, '*.jpg'))
	d = (size[0] * size[1])

	#design matrices, color stacks RBG
	X_color = np.zeros((N, d*3))
	X_gray = np.zeros((N, d))

	idx = 0
	for filename in sorted(dataset):
		sys.stderr.write("Processing: %s\n" % filename)

		#load image and resize
		im = Image.open(os.path.join(base_dir, filename))
		im = im.resize(size, Image.ANTIALIAS)

		#reshape to 1-d vector (and convert to grayscale)
		if(im.mode != 'RGB'): #ensure that any non RBG are converted (ie grayscale, CMYK)
			im = im.convert('RGB')
		color_array = np.array(im).ravel()
		gray_array = np.array(im.convert('L')).ravel()
		
		X_color[idx, :] = color_array.T
		X_gray[idx, :] = gray_array.T

		idx += 1

	print(idx)
	return X_color, X_gray

def read_labels(base_dir, label_mapping, filename):

	
	unique_labels = list(set(label_mapping.values()))
	
	class_labels = {} #map of class label to assocaited integer value
	for i in range(len(unique_labels)):
		class_labels[unique_labels[i]] = i


	#write class labels for future use
	w = csv.writer(open(filename + ".csv", "w"))
	for key, val in class_labels.items():
		w.writerow([key, val])




	image_filenames = glob.glob1(base_dir, '*.jpg')
	mapping_filenames = label_mapping.keys()
	dataset = intersect(image_filenames, mapping_filenames)

	#design matrices, color stacks RBG
	N = len(dataset)
	y = -1*np.ones((N))

	idx = 0
	for img in sorted(dataset):
		label = label_mapping[img]
		y[i] = class_labels[label]
		idx += 1

	return y
	

def write_dataset(X_color_style, X_gray_style, X_color_genre, X_gray_genre,\
	X_color_date, X_gray_date, y_style, y_genre, y_date):
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
	h5f.create_dataset('color_style', data=X_color_style)
	h5f.create_dataset('gray_style', data=X_gray_style)
	#h5f.create_dataset('color_genre', data=X_color_genre)
	#h5f.create_dataset('gray_genre', data=X_gray_genre)
	#h5f.create_dataset('color_date', data=X_color_date)
	#h5f.create_dataset('gray_date', data=X_gray_date)

	h5f.create_dataset('style', data=y_style)
	#h5f.create_dataset('genre', data=y_genre)
	#h5f.create_dataset('date', data=y_date)
	h5f.close()



def main():

	#resized image dimensions
	#TODO: square okay?
	size = 64, 64

	#base directory
	base_dir = "images/"
 
	#label_mapping('train_artist.csv', artist_labels) #not sure if this will be useful
	label_mapping('train_style.csv', style_labels)
	#label_mapping('train_genre.csv', genre_labels)
	#label_mapping('train_date.csv', date_labels)

	
	X_color_style, X_gray_style = process_images(base_dir, size, style_labels)
	#X_color_genre, X_gray_genre = process_images(base_dir, size, genre_labels)
	#X_color_date, X_gray_date = process_images(base_dir, size, date_labels)
	y_style= read_labels(base_dir, style_labels, 'labels_style')
	#y_genre = read_labels(base_dir, genre_labels, 'labels_genre')
	#y_date = read_labels(base_dir, date_labels, 'labels_date')


	#write_dataset(X_color_style, X_gray_style, X_color_genre, X_gray_genre, \
	#	X_color_date, X_gray_date, y_style, y_genre, y_date)

	write_dataset(X_color_style, X_gray_style, None, None, \
		None, None, y_style, None, None)


if __name__ == '__main__':
	main()