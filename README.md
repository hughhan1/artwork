# Artwork

Classification of Artwork Genres Using Image Data.

#### TODO
- Resize image (`.jpg`) files into a consistent M x N dimensionality format.
- Design classification labels. (Perhaps we can use year intervals for now.)
- Load data vectors and true labels into SciKit-Learn (`sklearn`), and begin training.

## Setup

First, clone this repository and `cd` into it.
```
$ git clone https://github.com/hughhan1/artwork.git
$ cd artwork
```

Next, create a virtual environment and install the necessary `pip` packages.
```
$ virtualenv env
$ source env/bin/activate
$ pip install -r requirements.txt
```

Before we do anything further, we need to retrieve our data. This data is available from
MoMA's GitHub repository in a JSON format, available 
[here](https://github.com/MuseumofModernArt/collection/blob/master/Artworks.json).
Save this file as `json/artworks.json`.

## Scraping

To scrape images from the MoMA collection, run the following command.
```
$ python moma.py
```

The `-t` or `--thumbnails` option can be provided to obtain thumbnails instead of
larger images, as shown below.
```
$ python moma.py -t
```

Regular sized images will be saved into the `images/` directory, and thumbnails will be
saved into the `thumbnails/` directory.