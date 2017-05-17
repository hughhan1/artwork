#!usr/bin/env python
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
import sklearn.metrics
import time
from datetime import timedelta
import math
import dataset
import random
import os
    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Convolutional Layer 1.
filter_size1 = 3 
num_filters1 = 32

# Convolutional Layer 2.
filter_size2 = 3
num_filters2 = 32

# Convolutional Layer 3.
filter_size3 = 3
num_filters3 = 64
    
# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 3

# image dimensions (only squares for now)
img_size = 128

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# class info

#classes = ['Photograph', 'Installation', 'Sculpture', 
#        'Illustrated Book', 'Design', 'Architecture', 
#        'Periodical', 'Print', 'Video', 'Painting', 'Drawing', 'Film']
#classes = ['Russian', 'Spanish', 'Mexican', 'Canadian', 'German', 'Brazilian', 'Japanese', 'French', 'Czech',
#        'American', 'British', 'Dutch', 'Swiss', 'Austrian', 'Italian', 'Colombian', 'Argentine', 'Belgian']

#classes = ['1976', '2003', '1954', '1961', '1921', '1923', '1924', '1962', '1926', '1927', '1928', '1929',
#    '1989', '1986', '1987', '1984', '1949', '1982', '1969', '1980', '1981', '1964', '1965', '1966', '1967',
#    '1960', '1947', '1988', '1963', '2001', '1985', '2011', '2004', '1978', '2005', '1948', '1933', '1932',
#    '1931', '1956', '1937', '1950', '1953', '1934', '1968', '1938', '1959', '1958', '1991', '1990', '1993',
#    '1992', '1995', '1994', '1997', '1996', '1977', '1998', '1975', '1974', '1973', '1972', '1971', '1970',
#    '2000', '1930', '2002', '1999', '2006', '2007', '1957', '1979', '1951', '2008', '2009', '1925', '1983']

#classes = ['1950-1960', '1990-2000', '1930-1930', '1970-1980', '1940-1950', '1950-1950' '1930-1940',
#     '1980-1980', '1920-1930', '1960-1960', '1990-1990', '1970-1970', '2010-2020', '2000-2010',
#     '2000-2000', '1960-1970', '1980-1990', '1930-1940', '1950-1950']

classes = [ 'Impressionism',                #8220
    'Realism',                      #8112
    'Romanticism',                  #7041
    'Expressionism',                #5325
    'Post-Impressionism']            #4527]
num_classes = len(classes)

# batch size
batch_size = 256

# validation split
validation_size = .2

# how long to wait after validation loss stops improving before terminating training
early_stopping = None  # use None if you don't want to implement early stoping


#tf.train.ClusterSpec({"local":["localhost:2222", "localhost:2222", "localhost:2222", "localhost:2222"]})

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(
    input,               # The previous layer.
    num_input_channels,  # Num. channels in prev. layer.
    filter_size,         # Width and height of each filter.
    num_filters,         # Number of filters.
    use_pooling=True     # Use 2x2 max-pooling.
):
    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(
        input=input,
        filter=weights,
        strides=[1, 1, 1, 1],
        padding='SAME'
    )

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(
            value=layer,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME'
        )

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


def new_fc_layer(
    input,          # The previous layer.
    num_inputs,     # Num. inputs from prev. layer.
    num_outputs,    # Num. outputs.
    use_relu=True   # Use Rectified Linear Unit (ReLU)?
):

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

def new_dropout_layer(
    input,
    rate,
    noise_shape,
    seed,
    training,
    name
):
    layer = tf.nn.dropout(input, rate, noise_shape, seed, name)
    return layer

def get_confusion_matrix(feed_dict_train, feed_dict_validate):

    predictions = session.run(y_pred_cls, feed_dict=feed_dict_validate)
    true        = session.run(y_true_cls, feed_dict=feed_dict_validate)

    # print("predicted: {0}".format(predictions))
    # print("true: {0}".format(true))

    return sklearn.metrics.confusion_matrix(
        y_true=true, 
        y_pred=predictions, 
        labels=list(range(len(classes)))
    )

def print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):

    # First, calculate the accuracy on the training set and the validation set.
    acc     = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)

    # Print the training and validation accuracy.
    print(
        "Epoch {0} --- Training Accuracy: {1:>6.1%}, "
        "Validation Accuracy: {2:>6.1%}, "
        "Validation Loss: {3:.3f}"
        .format(epoch + 1, acc, val_acc, val_loss)
    )

    print(get_confusion_matrix(feed_dict_train, feed_dict_validate))

    end_time = time.time()
    print("Time elapsed: %d" % (end_time - start_time))

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    best_val_loss = float("inf")

    # First, initialize a confusion matrix of 0s.
    # confusion_matrix = np.zeros((num_classes, num_classes))

    for i in range(total_iterations, total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)
       
        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, flattened image shape]
        x_batch = x_batch.reshape(train_batch_size, img_size_flat)
        x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)
        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        
        feed_dict_validate = {x: x_valid_batch,
                              y_true: y_valid_batch}

        #print(feed_dict_validate[x].shape)

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Get the confusion matrix for the current batch, and add it to the total.
        confusion_matrix = get_confusion_matrix(feed_dict_train, feed_dict_validate)

        # Print the confusion matrix for the current batch.
        # print(confusion_matrix)
        
        # Print status at end of each epoch (defined as full pass through training dataset).
        if i % int(data.train.num_examples/batch_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_validate)
            epoch = int(i / int(data.train.num_examples/batch_size))
            
            print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)

    # Update the total number of iterations performed.
    total_iterations += num_iterations

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement=True
config.intra_op_parallelism_threads = 16
session = tf.Session(config=config)
train_path='training_data'
test_path='testing_data'

#dataset.partition_train_test('moma_class.csv', 0.75)
#dataset.partition_train_test('moma_nation.csv', 0.75)
#dataset.partition_train_test('moma_date.csv', 0.75)
#dataset.partition_train_test('moma_start_date.csv', 0.75)
dataset.partition_train_test('train_style.csv', 0.75)

print('=====Reading Training Sets=====')
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
print('=====Reading Test Sets=====')
test_images, test_ids = dataset.read_test_set(test_path, img_size,classes)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(test_images)))
print("- Validation-set:\t{}".format(len(data.valid.labels)))

start_time = time.time()

#session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

#layer_conv1, weights_conv1 = \
#new_conv_layer(input=x_image,
#               num_input_channels=num_channels,
#               filter_size=filter_size1,
#               num_filters=num_filters1,
#               use_pooling=True)
#print("now layer2 input")
#print(layer_conv1.get_shape())     
#layer_conv2, weights_conv2 = \
#new_conv_layer(input=layer_conv1,
#               num_input_channels=num_filters1,
#               filter_size=filter_size2,
#               num_filters=num_filters2,
#               use_pooling=True)
#print("now layer3 input")
#print(layer_conv2.get_shape())     
               
#layer_conv3, weights_conv3 = \
#new_conv_layer(input=layer_conv2,
#               num_input_channels=num_filters2,
#               filter_size=filter_size3,
#               num_filters=num_filters3,
#               use_pooling=True)
#print("now layer flatten input")
#print(layer_conv3.get_shape())     
          
layer_flat, num_features = flatten_layer(x_image)#layer_conv1)

layer_fc1 = new_fc_layer(input=layer_flat,
                     num_inputs=num_features,
                     num_outputs=fc_size,
                     use_relu=True)
layer_dropout1 = new_dropout_layer(layer_fc1,
                    rate = 0.25,
                    noise_shape=None,
                    seed=None,
                    training=None,
                    name=None)
layer_fc2 = new_fc_layer(input=layer_dropout1,
                     num_inputs=fc_size,
                     num_outputs=num_classes,
                     use_relu=False)

y_pred = tf.nn.softmax(layer_fc2)

y_pred_cls = tf.argmax(y_pred, dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                    labels=y_true)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# confusion_matrix = tf.confusion_matrix

#session.run(tf.global_variables_initializer()) # for newer versions
session.run(tf.initialize_all_variables()) # for older versions


train_batch_size = batch_size

total_iterations = 0

    
optimize(num_iterations=3000)
#print_validation_accuracy()
