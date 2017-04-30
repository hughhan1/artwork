"""
This script downloads images from the Getty collection
"""


from bs4 import BeautifulSoup
from urllib  import urlretrieve
from urllib2 import urlopen

import csv
import json
import os
import sys


image_dir     = 'images'
thumb_dir     = 'thumbnails'

padding       = 6

def make_soup(url):
    """
    Instantiates and returns a BeautifulSoup instance using the URL provided in
    the function parameters.

    Args:
        url : the URL that the BeautifulSoup instance will use

    Returns:
        a BeautifulSoup instance using the URL provided
    """
    html = urlopen(url).read()
    return BeautifulSoup(html, 'lxml')


def get_image(url, filename):
    """
    Retrieves the image associated with the URL provided, and writes it using
    the filename provided.

    Args:
        url      : the URL with information about a particular piece of artwork
        filename : the filename to which the image should be saved

    Returns:
        a string containing the href link to the image that was written
    """
    
    soup       = make_soup(url)
    image_tag  = soup.find('meta', property='og:image', content=True)
    class_tag  = soup.find('meta', attrs={'name' : 'object_type'}, content=True)

    image_link     = ""
    classification = ""

    try:
        image_link     = image_tag['content']
        classification = class_tag['content']

        if image_link == "":
            print('error   : no image found at .' % (url))
        else:
            try:
                urlretrieve(image_link, os.path.join(image_dir, filename))
                print('success : %s downloaded to %s directory.' % (filename, image_dir))
            except AttributeError:
                print('error   : %s could not be downloaded to %s directory.' % (filename, image_dir))
    except TypeError:
        print('error   : %s could not be downloaded to %s directory.' % (filename, image_dir))

    return image_link, classification


def get_images(ids):
    """
    Iterates through a ID numbers on getty.edu, and retrieves all images associated with
    it, and downloads them to an image directory

    Args:
        ids : list of artwork IDs (numeric)
    """

    classification_labels = {}

    for id in range(1, int(ids)+1):

        # Iterate through all of the desired IDs, and obtain images and
        # classifications of the artwork
        url = "https://www.getty.edu/art/collection/objects/" + str(id)
        image_filename = 'getty_' + str(id).zfill(padding) + '.jpg'
        _, classification = get_image(url, image_filename)

        # If the classification is an empty string, that means that there was
        # some error with either obtaining the actual artwork itself or the
        # classification of the artwork.
        if classification != "":
            classification_labels['getty_' + str(id)] = classification

    write_labels(classification_labels, 'getty.csv')


def write_labels(labels_map, filename):
    """
    Creates CSV file of labels (nationality, date) to be used by utils.py

    Args:
        labels_map : a dictionary with key objectID and value label name
        filename   : the filename to which the output should be written
    """

    unique_values = set()            # Create a set of non-repeated labels
    for val in labels_map.values():  # from our label map.
        unique_values.add(val)

    sys.stderr.write("======== Statistics ========\n")
    for val in unique_values:        # Count the number of times each label
        sys.stderr.write(            # occured.
            "%s : %d\n" % 
            (val, sum(x == val for x in labels_map.values()))
        )
    sys.stderr.write("============================\n")

    # Write the results to a CSV file, structured as objectID\tlabel
    with open(filename, "w") as output:
        writer = csv.writer(output, delimiter='\t', lineterminator='\n')
        for key, val in labels_map.iteritems():
            writer.writerow([key, val])


def main():
    get_images(sys.argv[1])


if __name__ == '__main__':
    main()
