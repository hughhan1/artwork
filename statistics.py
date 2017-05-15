import os, sys
import numpy as np
import csv
import re

artist_labels = {}
style_labels = {}
genre_labels = {}
date_labels = {} #global so can be used elsewhere


def intersect(a, b):
	return list(set(a) & set(b))


def label_mapping(filename):
	"""
	Creates mapping of filename to label from CSV

	Args:
		filename	: name of dataset CSV file
	"""

	with open(filename, 'r') as infile:
		reader = csv.reader(infile)
		next(reader, None) # ignore first line since they're column labels

		#filename, artist, title, style, genre, date
		for line in reader:
			img = line[0]
			artist = line[1]
			style = line[3]
			genre = line[4]
			date = re.findall(r'\d+', line[5]) #parse any unwanted stuff

			#img and artist fields always present, no need to check
			artist_labels[img] = artist


			if style != '':
				style_labels[img] = style
			if genre != '':
				genre_labels[img] = genre
			if len(date) > 0:
				date_labels[img] = date[0] #parsed_date


def label_stats(label_mapping):
	"""
    Determines the number of exampes for each label, sorted from hightest to lowest

    Args:
        label_mapping : the dictionary label mapping to inspect
    """
	labels = list(label_mapping.values())

	for count, elem in sorted(((labels.count(e), e) for e in set(labels)), reverse=True):
		print('%s: \t\t %d' % (elem, count))

def main():

	label_mapping('train_info.csv')

	'''
	artist_labels
	style_labels
	genre_labels
	date_labels
	'''
	label_stats(style_labels)
	label_stats(genre_labels)

if __name__ == '__main__':
	main()