# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from tensorflow import logging

import os

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import time


logging.set_verbosity(logging.INFO)

tf.app.flags.DEFINE_integer('training_iteration', 1000,
                            'Number of training iterations')
tf.app.flags.DEFINE_integer('model_version', 1, 'Version number of the model')
tf.app.flags.DEFINE_string('build', '1' , 'Build for export')
tf.app.flags.DEFINE_string('catalog_name', None , 'Catalog name')
tf.app.flags.DEFINE_string('data_dir', '/tmp/mnist/data' , 'Data directory')
tf.app.flags.DEFINE_string('log_dir', '/tmp/mnist/training', 'Log directory')
tf.app.flags.DEFINE_string('source_url', 'https://storage.googleapis.com/cvdf-datasets/mnist/','source')
tf.app.flags.DEFINE_integer('fully_neurons', 3, 'Number of fully connected neurons')
tf.app.flags.DEFINE_float('drop_out', 0.5, 'Drop out')
FLAGS = tf.app.flags.FLAGS

def deepnn(x):
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    keep_prob = tf.placeholder(tf.float32)
    return deepnn_builder(x_image,keep_prob)

def deepnn_builder(x_image,keep_prob):
    """deepnn builds the graph for a deep net for classifying digits.
    Args:
      x: an input tensor with the dimensions (N_examples, 784), where 784 is the
      number of pixels in a standard MNIST image.
    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the digit into one of 10 classes (the
      digits 0-9). keep_prob is a scalar placeholder for the probability of
      dropout.mnist-sm-build-27
    """
    # First convolutional layer - maps one grayscale image to 32 feature maps.
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    n = FLAGS.fully_neurons*256+256
    W_fc1 = weight_variable([7 * 7 * 64, n])
    b_fc1 = bias_variable([n])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    W_fc2 = weight_variable([n, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    train()



def train():
    # Import data
    logging.info('Loading dataset...')
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, source_url=FLAGS.source_url)

    logging.info('Initing tf graph...')
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784],name="x")

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10],name="y_")



    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x)
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        start = time.time()
        for i in range(FLAGS.training_iteration):
            batch = mnist.train.next_batch(50)
            _ = sess.run([train_step],feed_dict={x: batch[0], y_: batch[1], keep_prob: FLAGS.drop_out})
            if i % 100 == 0 and i > 0:
                delta = time.time() - start
                logging.info('Step %d,  %.2f steps/s',i, 100.0/delta)

        saver.save(sess,os.path.join(FLAGS.log_dir,"model.ckpt"))
        test_accuracy = accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        logging.info('Test accuracy %g',test_accuracy)



if __name__ == '__main__':
    tf.app.run()
