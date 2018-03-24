from tensorflow import logging
import os
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from mlboardclient.api import client

logging.set_verbosity(logging.INFO)
tf.app.flags.DEFINE_string('catalog_name', 'mnist', 'Catalog name')
tf.app.flags.DEFINE_string('version', '1.0.'+os.environ.get('BUILD_ID'), 'Dataset version')
tf.app.flags.DEFINE_string('source_url', 'https://storage.googleapis.com/cvdf-datasets/mnist/', 'Source url')
FLAGS = tf.app.flags.FLAGS

def main(_):
    data_dir = "/tmp/mnist"
    logging.info("Uploading dataset to %s ...",data_dir)
    input_data.read_data_sets(data_dir, one_hot=True, source_url=FLAGS.source_url)
    logging.info("Upload success")
    logging.info("Pushing dataset to %s:%s ...",FLAGS.catalog_name,FLAGS.version)
    kl = client.Client()
    kl.datasets.push(os.environ.get('WORKSPACE_NAME'),FLAGS.catalog_name,FLAGS.version,data_dir,create=True)
    logging.info("Push success")

if __name__ == '__main__':
    tf.app.run()
