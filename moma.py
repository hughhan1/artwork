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
import re
import socket
import sys

from interval import Interval


image_dir     = 'images'
thumb_dir     = 'thumbnails'
artworks_file = 'json/artworks.json'

padding       = 6

max_imgs = 100000 #TODO: pass as argument


nation_check = [
    'Argentine',
    'Belgian',
    'French',
    'Canadian',
    'Italian',
    'German',
    'Austrian',
    'Brazilian',
    'Russian',
    'American',
    'Czech',
    'Dutch',
    'Colombian',
    'British',
    'Swiss',
    'Mexican',
    'Japanese',
    'Spanish'
]

class_check = [
    'Print',
    'Painting',
    'Illustrated Book',
    'Design',
    'Photograph',
    'Installation',
    'Architecture',
    'Sculpture',
    'Drawing',
    'Video',
    'Periodical',
    'Film'
]

date_check = [
'2007','1930','2011','1991','1990','1993','1992','1995','1994','1997','1996',
'1999','1998','1979','1978','1977','1976','1975','1972','1971','1970','1954',
'1957','1956','1951','1950','1953','1959','1958','1933','1932','1931','1937',
'1934','1938','2002','2003','2000','2001','2004','2005','2008','2009','1924',
'1986','1987','1984','1985','1982','1983','1980','1981','1988','1989','1974',
'1973','1968','1969','1964','1965','1966','1967','1960','1961','1962','1963',
'1948','1949','1947','1921','1923','1925','1926','1927','1928','1929', '2006',
]

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
    nation_labels = {}
    date_labels = {}
    start_date_labels = {}
    end_date_labels = {}
    time_period_labels = {}

    i = 0
    for artwork in artworks_data:

        url            = artwork['URL']
        object_id      = artwork['ObjectID']
        classification = artwork['Classification']
        nation         = artwork['Nationality']
        date           = artwork['Date']

        start_date, end_date = process_date(date)

        if url is not None and classification is not None and len(nation) != 0 and date is not None:
            if classification in class_check and date in date_check and nation[0] in nation_check:
            
                # Get the filename of the image using the Object ID of the image, and save the image
                # to disk.
                image_filename = 'moma_' + str(object_id).zfill(padding) + '.jpg'
                # get_image(url, image_filename)

                classification_labels['moma_' + str(object_id).zfill(padding)] = classification 
                nation_labels['moma_' + str(object_id).zfill(padding)] = nation[0]
                date_labels['moma_' + str(object_id).zfill(padding)]  = date

            if start_date is not None:
                start_date_labels['moma_' + str(object_id).zfill(padding)] = Interval.range_str(start_date)
            if end_date is not None:
                end_date_labels['moma_' + str(object_id).zfill(padding)]  = Interval.range_str(end_date)
            
            if start_date is not None:
                time_period_labels['moma_' + str(object_id).zfill(padding)] = Interval.time_period(start_date)

    write_labels(classification_labels, "moma_class.csv")
    write_labels(nation_labels, "moma_nation.csv")
    write_labels(date_labels, "moma_date.csv")
    write_labels(start_date_labels, "moma_start_date.csv")
    write_labels(end_date_labels, "moma_end_date.csv")
    write_labels(time_period_labels, "moma_time_period.csv")
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
        get_thumbnail(url, str(object_id).zfill(padding) + '.jpg')
    
    artworks_file.close()


def process_date(date):

    deletions = [
        'c.', 'arranged', 'published', 'fabricated', 'released', 'printed',
        'assembled', 'performed', '(', ')', 'January', 'February', 'March',
        'April', 'May', 'June', 'July', 'August', 'September', 'November',
        'December', 'spring', 'summer', 'fall', 'winter', 'late', 'or', 'n.d.',
    ]

    if date is None:
        return None, None
    
    date = ''.join(date.split())
    for deletion in deletions:
        date = date.replace(deletion, '')

    dates = re.split(',|-', date)
    start_date = dates[0]

    if len(dates) > 1:
        if len(dates[1]) == 4:
            end_date = dates[1]
        elif len(dates[1]) == 2:
            end_date = start_date[0:2] + date[1]
        elif dates[1] == 'present' or dates[1] == 'ongoing':
            end_date = '2017'
        else:
            end_date = None
    else:
        end_date = None
    
    try:
        start_date = int(start_date)
        if start_date < 1500 or start_date > 2017:
            start_date = None
    except ValueError:
        start_date = None
    except TypeError:
        start_date = None
    
    try:
        end_date = int(end_date)
        if end_date < 1500 or end_date > 2017:
            end_date = None
    except ValueError:
        end_date = None
    except TypeError:
        end_date = None

    return start_date, end_date


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

    sys.stderr.write('======== Statistics ========\n')
    for val in unique_values:        # Count the number of times each label
        sys.stderr.write(            # occured.
            "%s : %d\n" % 
            (val, sum(x == val for x in labels_map.values()))
        )
    sys.stderr.write('============================\n')

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

