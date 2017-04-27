"""
This script downloads images from the MoMA collection, using data provided in
the artworks.json file.
"""


from bs4 import BeautifulSoup
from urllib  import urlretrieve
from urllib2 import urlopen

import json
import os
import sys


image_dir     = 'getty_images'
thumb_dir     = 'getty_thumbnails'

#https://www.getty.edu/art/collection/objects/


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
            print(image_link)
            urlretrieve(image_link, os.path.join(image_dir, filename))
            print('success : %s downloaded to %s directory.' % (filename, image_dir))
        except AttributeError:
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


def get_images(ids):
    """
    Iterates through a ID numbers on getty.edu, and retrieves all images associated with
    it, and downloads them to an image directory

    Args:
        ids : list of artwork IDs (numeric)
    """
    for id in ids:
        url       = "https://www.getty.edu/art/collection/objects/" + str(id)
        print(url)
        object_id = id
        if url is not None:
            get_image(url, str(object_id) + '.jpg')

def get_thumbnails(ids):
    """
    Iterates through a ID numbers on getty.edu, and retrieves all thumbnails associated
    with it, and downloads them to a thumbnail directory

    Args:
        ids : the list of artwork IDs (numeric)
    """

    for id in ids:
        url       = "https://www.getty.edu/art/collection/objects/" + str(id)
        print(url)
        object_id = id
        get_thumbnail(url, str(object_id) + '.jpg')



def main():
    if len(sys.argv) > 2:
        if sys.argv[1] == '-t' or sys.argv[1] == '--thumbnails':
            get_thumbnails(sys.argv[2])
        else:
            raise Exception
    elif len(sys.argv) > 1:
        get_images(sys.argv[1])


if __name__ == '__main__':
    main()
