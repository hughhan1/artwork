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

processed_img = []


def intersect(a, b):
	return list(set(a) & set(b))


def label_mapping(filename):
	"""
	Creates mapping of filename to label from CSV

	Args:
		filename	: name of dataset CSV file
	"""

	with open(filename, 'rb') as infile:
		reader = csv.reader(infile)
		next(reader, None) # ignore first line since they're column labels

		#filename, artist, title, style, genre, date
		for line in reader:
			img = line[0]
			artist = line[1]
			style = line[3]
			genre = line[4]
			date = line[5]

			#img and artist fields always present, no need to check
			artist_labels[img] = artist

			if style != '':
				genre_labels[img] = genre
			if genre != '':
				genre_labels[img] = genre
			if date != '':
				date_labels[img] = date

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




	image_filenames = glob.glob1(base_dir, '*.jpg')
	mapping_filenames = labels.keys()
	dataset = intersect(image_filenames, mapping_filenames)

	idx = 0
	for img in sorted(dataset):
		label = label_mapping[img]
		y[i] = class_labels[label]
		idx += 1

	return y
	

def write_dataset(X_color_artist, X_gray_artist, X_color_genre, X_gray_genre,\
	X_color_date, X_gray_date, y_artist, y_genre, y_date):
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
	h5f.create_dataset('color_artist', data=X_color_artist)
	h5f.create_dataset('gray_artist', data=X_gray_artist)
	h5f.create_dataset('color_genre', data=X_color_genre)
	h5f.create_dataset('gray_genre', data=X_gray_genre)
	h5f.create_dataset('color_date', data=X_color_date)
	h5f.create_dataset('gray_date', data=X_gray_date)

	h5f.create_dataset('artist', data=y_artist)
	h5f.create_dataset('genre', data=y_genre)
	h5f.create_dataset('date', data=y_date)
	h5f.close()



def main():

	#resized image dimensions
	#TODO: square okay?
	size = 64, 64

	#base directory
	base_dir = "images/"
 
	label_mapping('train_info.csv')

	
	X_color_artist, X_gray_artist = process_images(base_dir, size, artist_labels)
	X_color_genre, X_gray_genre = process_images(base_dir, size, genre_labels)
	X_color_date, X_gray_date = process_images(base_dir, size, date_labels)
	y_artist = read_labels(artist_labels, 'artist_labels')
	y_genre = read_labels(genre_labels, 'genre_labels')
	y_date = read_labels(date_labels, 'date_labels')


	write_dataset(X_color_artist, X_gray_artist, X_color_genre, X_gray_genre, \
		X_color_date, X_gray_date, y_artist, y_genre, y_date)


	print(set(genre_labels.values()))

if __name__ == '__main__':
	main()