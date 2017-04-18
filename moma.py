"""
This script downloads images from the MoMA collection, using data provided in
the artworks.json file.
"""


from bs4 import BeautifulSoup
import json
from urllib2 import urlopen
import urllib


image_dir     = 'images'
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

    print('Downloading %s to %s directory.' % (filename, image_dir))
    urllib.urlretrieve(image_link, image_dir + '/' + filename)
    return image_link


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


def main():
    get_images(artworks_file)


if __name__ == '__main__':
    main()
