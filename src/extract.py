from tensorflow import logging
import os
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from mlboardclient.api import client
import scipy.misc
import tensorflow as tf

logging.set_verbosity(logging.INFO)
tf.app.flags.DEFINE_string('catalog_name', 'mnist', 'Catalog name')
tf.app.flags.DEFINE_string('version', '1.0.0-test', 'Dataset version')
tf.app.flags.DEFINE_string('data_dir', os.environ.get('DATA_DIR') , 'Data directory')
tf.app.flags.DEFINE_string('source_url', os.environ.get('DATA_DIR')+'/', 'Source url')
FLAGS = tf.app.flags.FLAGS

def main(_):
    img_dir = '/tmp/mnist'
    logging.info("Extracting images to %s ...",img_dir)
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, source_url=FLAGS.source_url)

    os.mkdir(img_dir)
    for i in range(mnist.test.num_examples):
        batch = mnist.test.next_batch(1)
        v = np.reshape(batch[0][0],(28,28))*255
        l = batch[1][0]
        l = l.astype(int)
        n = 0
        for k in l:
            if k==1:
                break
            n+=1
        l = ('%s/%d-%d.png') % (img_dir,i,n)
        scipy.misc.imsave(l, v)
    logging.info("Pushing dataset to %s:%s ...",FLAGS.catalog_name,FLAGS.version)
    kl = client.Client()
    kl.datasets.push(os.environ.get('WORKSPACE_NAME'),FLAGS.catalog_name,FLAGS.version,img_dir,create=True)
    logging.info("Push success")

if __name__ == '__main__':
    tf.app.run()
