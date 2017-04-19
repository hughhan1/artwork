import os, sys
import numpy as np
from PIL import Image
import h5py

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
	images = os.listdir(base_dir)

	#dimensions for design matrices
	#TODO: assumes that all images were able to be downloaded
	N = len(images)
	d = (size[0] * size[1])

	#design matrices, color stacks RBG
	X_color = np.zeros((N, d*3))
	X_gray = np.zeros((N, d))

	for file in images:
		print("Processing: " + file)

		#index into design matrix, can be arbitrary but we may want 
		#to have access to each specifc images' component later
		idx = int(os.path.splitext(file)[0]) - 1

		#load image and resize
		im = Image.open(base_dir+file)
	   	im = im.resize(size, Image.ANTIALIAS)

	   	#reshape to 1-d vector (and convert to grayscale)
		color_array = np.array(im).ravel()
		gray_array = np.array(im.convert('L')).ravel()

		#load 1-d vector into respective position
	   	X_color[idx, :] = color_array.T
	   	X_gray[idx, :] = gray_array.T

	return X_color, X_gray

def write_dataset(X_color, X_gray):
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
	h5f.close()



def main():
	#resized image dimensions
	#TODO: square okay?
	size = 500, 500

	#base directory
	base_dir = "images/"
	X_color, X_gray = process_images(base_dir, size)
	write_dataset(X_color, X_gray)


if __name__ == '__main__':
    main()