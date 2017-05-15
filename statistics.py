import os, sys
import numpy as np
import csv
import re

artist_labels = {}
style_labels = {}
genre_labels = {}
date_labels = {} #global so can be used elsewhere

# all have +600 examples
date_check = [
	'1910-1919', '1900-1909', '1890-1899', '1880-1889', '1920-1929', 
	'1930-1939', '1870-1879', '1960-1969',  '1950-1959', '1940-1949', 
	'1970-1979', '1860-1869', '1980-1989',  '1850-1859', '1990-1999',
	'1840-1849',  '1830-1839',  '1820-1829',  '1810-1819' 
]

genre_check = [						#number of examples
	'portrait', 		 			#12926
	'landscape', 		 			#11548
	'genre painting', 		 		#10984
	'abstract', 		 			#7201
	'religious painting', 		 	#5703
	'cityscape', 		 			#4089
	'sketch and study', 		 	#2778
	'illustration', 		 		#2493
	'still life', 		 			#2464
	'symbolic painting', 		 	#1959
	'figurative', 		 			#1782
	'nude painting (nu)', 		 	#1758
	'design', 		 				#1577
	'mythological painting', 		 #1493
	'marina', 		 				#1385
	'flower painting', 		 		#1270
	'animal painting', 		 		#1233
	'self-portrait', 		 		#1199
	'allegorical painting', 		 #809
	'history painting'				#656
]


style_check = [						#number of examples
	'Impressionism', 		 		#8220
	'Realism', 		 				#8112
	'Romanticism', 		 			#7041
	'Expressionism', 		 		#5325
	'Post-Impressionism', 		 	#4527
	'Art Nouveau (Modern)', 		#3779
	'Baroque', 						#3254
	'Surrealism', 					#3133
	'Symbolism', 		 			#2626
	'Rococo', 		 				#2101
	'Northern Renaissance', 		 #1824
	'Naïve Art (Primitivism)', 		# 1776
	'Neoclassicism', 		 		#1622
	'Abstract Expressionism', 		 #1546
	'Cubism', 		 				#1316
	'Ukiyo-e', 		 				#1137
	'Early Renaissance', 		 	#1052
	'High Renaissance', 		 	#1050
	'Mannerism (Late Renaissance)',	#1025
	'Art Informel', 		 		#987
	'Academicism', 		 			#766
	'Abstract Art', 		 		#759
	'Magic Realism', 				#696
	'Color Field Painting', 		#675
	'Pop Art', 		 				#619

]


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
				bucket_len = 10 #buckets of 10 years
				bucket = (int(date[0]) // bucket_len) * bucket_len 
				date_labels[img] = str(bucket) + '-' + str(bucket + (bucket_len - 1)) #parsed_date


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
	print('STYLE:\n')
	label_stats(style_labels)
	print('=================================================')
	print('GENRE:\n')
	label_stats(genre_labels)
	print('=================================================')
	print('DATES:\n')
	label_stats(date_labels)
	print('=================================================')

if __name__ == '__main__':
	main()