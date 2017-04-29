"""
This script downloads images from the MoMA collection, using data provided in
the artworks.json file.
"""


from bs4 import BeautifulSoup
from urllib  import urlretrieve
from urllib2 import urlopen

import csv
import json
import os
import socket
import sys


image_dir     = 'images'
thumb_dir     = 'thumbnails'
artworks_file = 'json/artworks.json'

padding       = 6

max_imgs = 10000 #TODO: pass as argument


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
    image_link = image_tag['content']


    if image_link == "":
        print('error   : no image found at .' % (url))
    else:
        try:
            request = urlopen(image_link, timeout=5)
            with open(os.path.join(image_dir, filename), 'w') as f:
                f.write(request.read())
            print('success : %s downloaded to %s directory.' % (filename, image_dir))
        except AttributeError:
            print('error   : %s could not be downloaded to %s directory.' % (filename, image_dir))
        except socket.timeout:
            print('error   : %s could not be downloaded to %s directory.' % (filename, image_dir))
    return image_link


def get_thumbnail(url, filename):
    """
    Retrieves the thumbnail associated with the URL provided, and writes it
    using the filename provided.

    Args:
        url      : the URL with information about a particular piece of artwork
        filename : the filename to which the thumbnail should be saved

    Returns:
        a string containing the href link to the thumbnail that was written
    """

    if url == "":
        print('error: url cannot be an empty string.' % (filename, thumb_dir))
    else:
        try:
            urlretrieve(url, os.path.join(thumb_dir, filename))
            print('success : %s downloaded to %s directory.' % (filename, thumb_dir))
        except AttributeError:
            print('error   : %s could not be downloaded to %s directory.' % (filename, thumb_dir))
    return url


def get_images(artworks_filename):
    """
    Iterates through a JSON data file, and retrieves all images associated with
    it, and downloads them to an image directory

    Args:
        artworks_filename : the filename of the JSON data file
    """
    artworks_file = open(artworks_filename)
    artworks_data = json.load(artworks_file)

    classification_labels = {}
    i = 0
    for artwork in artworks_data[:max_imgs]:

        if i % 500 != 0:
            i += 1
            continue
        else:
            i += 1

        url            = artwork['URL']
        object_id      = artwork['ObjectID']
        classification = artwork['Classification']

        if url is not None:
            image_filename = 'moma_' + str(object_id).zfill(padding) + '.jpg'
            get_image(url, image_filename)
            classification_labels['moma_' + str(object_id)] = classification 
    
    write_labels(classification_labels, "moma.csv")
    artworks_file.close()


def get_thumbnails(artworks_filename):
    """
    Iterates through a JSON data file, and retrieves all thumbnails associated
    with it, and downloads them to a thumbnail directory

    Args:
        artworks_filename : the filename of the JSON data file
    """
    artworks_file = open(artworks_filename)
    artworks_data = json.load(artworks_file)
    
    for artwork in artworks_data:
        url       = artwork['ThumbnailURL']
        object_id = artwork['ObjectID']
        get_thumbnail(url, str(object_id).zfill(pad) + '.jpg')
    
    artworks_file.close()


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

    for val in unique_values:        # Count the number of times each label
        sys.stderr.write(            # occured.
            "%s : %d\n" % 
            (val, sum(x == val for x in labels_map.values()))
        )

    w = csv.writer(open(filename, "w"))
    for key, val in labels_map.items():
        w.writerow([key, val])


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == '-t' or sys.argv[1] == '--thumbnails':    
            get_thumbnails(artworks_file)
        else:
            raise Exception
    get_images(artworks_file)


if __name__ == '__main__':
    main()
