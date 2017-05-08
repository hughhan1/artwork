import os
import os.path
import glob
import numpy as np
import cv2
import random
from sklearn.utils import shuffle

def partition_train_test(fn,p):
    #p is probability that an element is in the training set 
    training = {}
    test = {}
    with open(fn, 'r') as f:
      lines = f.readlines()
      #print(len(lines))
      i=0
      for line in shuffle(lines):
          if i % 40 == 0:
            if random.random() > p:
                #linesplit = line.split(',')
                #print("Checking image: " + linesplit[0])
                if os.path.isfile('images/' + line.split(',')[0] + '.jpg'):
                    test[line.split(',')[0]] = line.split(',')[1]
            else:
                if os.path.isfile('images/' + line.split(',')[0] + '.jpg'):
                    training[line.split(',')[0]] = line.split(',')[1]
          i+=1
    with open('test_set.csv', 'w') as test_file:
        for key in test:
            test_file.write(key + ',' + test[key])
    with open('training_set.csv', 'w') as training_file:
        for key in training:
             training_file.write(key + ',' + training[key])
    #test_file = open('test_set.csv', 'w')
    #for key in test:
    #    test_file.write(key + ',' + test[key])    
    #training_file = open('training_set.csv', 'w')
    #for key in training:
    #    training_file.write(key + ',' + training[key])

def load_train(train_path, image_size, classes):
    images = []
    labels = []
    ids = []
    cls = []
    training_dict = {}
    print('Reading training images')
    #for fld in classes:   # assuming data directory has a separate folder for each class, and that each folder is named after the class
    #    index = classes.index(fld)
    #    print('Loading {} files (Index: {})'.format(fld, index))
    #    path = os.path.join(train_path, fld, '*g')
    #    files = glob.glob(path)
    files = []
    with open('training_set.csv', 'r') as training_file:
        lines = training_file.readlines()
        for line in lines:
            files.append(line.split(',')[0])
            ids.append(line.split(',')[0])
            cls.append(line.split(',')[1])
            training_dict[line.split(',')[0]] = line.split(',')[1].rstrip()
        for fl in files:
            try:
                print('Processing training set image: ' + fl)
                image = cv2.imread('images/' + fl + '.jpg')
                image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
                images.append(image)
                label = np.zeros(len(classes))
                index = classes.index(training_dict[fl])
                label[index] = 1.0
                labels.append(label)
                flbase = os.path.basename(fl)
            except IOError:
                continue
            #ids.append(flbase)
            #cls.append(fld)
    images = np.array(images)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)

    return images, labels, ids, cls


def load_test(test_path, image_size,classes):
    test_dict = {}
    classes = []
    with open('test_set.csv', 'r') as test_file:
        lines = test_file.readlines()
        for line in lines:
            test_dict[line.split(',')[0]] = line.split(',')[1]
            if line.split(',')[1] not in classes:
                classes.append(line.split(',')[1])
    #for class_name in classes:
    #    path = os.path.join(test_path,class_name, '*g')
    #    files = sorted(glob.glob(path))
    for class_name in classes:
        X_test = []
        X_test_id = []
        print("Reading test images")
        #for fl in files:
            #flbase = os.path.basename(fl)
            #print(fl)
        for f1 in test_dict:
            try:
                print("Processing test image: " + f1)
                img = cv2.imread('images/' + f1 + '.jpg')
                img = cv2.resize(img, (image_size, image_size), cv2.INTER_LINEAR)
                X_test.append(img)
                X_test_id.append(test_dict[f1])
            except IOError:
                continue
  ### because we're not creating a DataSet object for the test images, normalization happens here
        X_test = np.array(X_test, dtype=np.uint8)
        X_test = X_test.astype('float32')
        X_test = X_test / 255

    return X_test, X_test_id



class DataSet(object):

  def __init__(self, images, labels, ids, cls):
    self._num_examples = images.shape[0]
    images = images.astype(np.float32)
    images = np.multiply(images, 1.0 / 255.0)

    self._images = images
    self._labels = labels
    self._ids = ids
    self._cls = cls
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def ids(self):
    return self._ids

  @property
  def cls(self):
    return self._cls

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1

      # # Shuffle the data (maybe)
      # perm = np.arange(self._num_examples)
      # np.random.shuffle(perm)
      # self._images = self._images[perm]
      # self._labels = self._labels[perm]
      # Start next epoch

      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end], self._ids[start:end], self._cls[start:end]


def read_train_sets(train_path, image_size, classes, validation_size=0):
  class DataSets(object):
    pass
  data_sets = DataSets()

  images, labels, ids, cls = load_train(train_path, image_size, classes)
  images, labels, ids, cls = shuffle(images, labels, ids, cls)  # shuffle the data

  if isinstance(validation_size, float):
    validation_size = int(validation_size * images.shape[0])

  validation_images = images[:validation_size]
  validation_labels = labels[:validation_size]
  validation_ids = ids[:validation_size]
  validation_cls = cls[:validation_size]

  train_images = images[validation_size:]
  train_labels = labels[validation_size:]
  train_ids = ids[validation_size:]
  train_cls = cls[validation_size:]

  data_sets.train = DataSet(train_images, train_labels, train_ids, train_cls)
  data_sets.valid = DataSet(validation_images, validation_labels, validation_ids, validation_cls)

  return data_sets


def read_test_set(test_path, image_size,classes):
  images, ids  = load_test(test_path, image_size,classes)
  return images, ids