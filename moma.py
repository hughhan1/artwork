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


image_dir     = 'images'
thumb_dir     = 'thumbnails'
artworks_file = 'json/artworks.json'


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
            urlretrieve(image_link, os.path.join(image_dir, filename))
            print('success : %s downdloaded to %s directory.' % (filename, image_dir))
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
            print('success : %s downdloaded to %s directory.' % (filename, thumb_dir))
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
    
    for artwork in artworks_data:
        url       = artwork['URL']
        object_id = artwork['ObjectID']
        get_image(url, str(object_id) + '.jpg')
    
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
        get_thumbnail(url, str(object_id) + '.jpg')
    
    artworks_file.close()


def main():
    if sys.argv > 1:
        if sys.argv[1] == '-t' or sys.argv[1] == '--thumbnails':    
            get_thumbnails(artworks_file)
        else:
            raise Exception
    get_images(artworks_file)


if __name__ == '__main__':
    main()
