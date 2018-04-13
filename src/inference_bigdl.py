import argparse
import logging
import os.path as op
import os
from os import path

from bigdl.util import common
from bigdl.nn import layer
from bigdl.util import tf_utils
import numpy as np
from PIL import Image
import pyspark
from six.moves import StringIO
import tensorflow as tf


logging.basicConfig(
    format='%(asctime)s %(levelname)-10s %(name)-25s [-] %(message)s',
    level='INFO'
)
logging.root.setLevel(logging.INFO)
LOG = logging.getLogger('train')


def export_tf_checkpoint(*args):
    meta_file = None
    checkpoint = None
    save_path = "model"
    saver_folder = None

    if len(args) == 1:
        if op.isdir(args[0]):
            saver_folder = args[0]
        else:
            meta_file = args[0] + ".meta"
            checkpoint = args[0]
    elif len(args) == 2:
        if op.isdir(args[0]):
            saver_folder = args[0]
        else:
            meta_file = args[0] + ".meta"
            checkpoint = args[0]
        save_path = args[1]
    elif len(args) == 3:
        meta_file = args[0]
        checkpoint = args[1]
        save_path = args[2]
    else:
        print("Invalid script arguments. How to run the script:\n" +
              "python export_tf_checkpoint.py checkpoint_name\n" +
              "python export_tf_checkpoint.py saver_folder\n" +
              "python export_tf_checkpoint.py checkpoint_name save_path\n" +
              "python export_tf_checkpoint.py saver_folder save_path\n" +
              "python export_tf_checkpoint.py meta_file checkpoint_name save_path")
        exit(1)

    if op.isfile(save_path):
        print("The save folder is a file. Exit")
        exit(1)

    if not op.exists(save_path):
        print("create folder " + save_path)
        os.makedirs(save_path)

    with tf.Session() as sess:
        if saver_folder is None:
            saver = tf.train.import_meta_graph(meta_file, clear_devices=True)
            saver.restore(sess, checkpoint)
        else:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saver_folder)
            checkpoint = save_path + '/model.ckpt'
            saver = tf.train.Saver()
            saver.save(sess, checkpoint)
        tf_utils.dump_model(save_path, None, sess, checkpoint)


def load(spark_files):
    data = []
    for f in spark_files:
        with open(pyspark.SparkFiles.get(path.basename(f))) as file:
            # Add tuples (path, data)
            data.append((path.basename(f), load_input(file.read())))

    return data


def load_input(data):
    image = Image.open(StringIO(data))
    im_width, im_height = image.size
    return np.array(image.getdata()).reshape(
        (1, im_height * im_width)).astype(np.uint8)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ckpt-prefix',
    )
    parser.add_argument(
        '--model-path',
    )
    parser.add_argument(
        '--input',
        help='input image path.'
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    conf = (
        common.create_spark_conf()
        .setAppName('pyspark-mnist')
        # .setMaster(args.master)
    )
    conf = conf.set('spark.executor.cores', 4)
    conf = conf.set('spark.cores.max', 4)
    LOG.info('initialize with spark conf:')
    LOG.info(conf.getAll())
    sc = pyspark.SparkContext(conf=conf)
    # sc.setLogLevel('INFO')
    common.init_engine()

    if not args.model_path and not args.ckpt_prefix:
        raise RuntimeError('provide either --ckpt-prefix or --model-path')

    bin_file = None
    if args.ckpt_prefix:
        save_path = '/tmp/model'
        export_tf_checkpoint(args.ckpt_prefix, save_path)
        model = save_path + '/model.pb'
        bin_file = save_path + '/model.bin'
    else:
        model = args.model_path

    inputs = ['x']
    outputs = ['y_']

    LOG.info('Loading model...')
    bigdl_model = layer.Model.load_tensorflow(
        model,
        inputs,
        outputs,
        bin_file=bin_file,
    )
    LOG.info('Loading inputs...')

    if path.isdir(args.input):
        files = [path.join(args.input, f) for f in os.listdir(args.input)]
    else:
        files = [args.input]

    for f in files:
        # Add all files to spark
        sc.addFile(f)

    # Load raw data into numpy arrays
    images = sc.parallelize(files).mapPartitions(load)

    LOG.info('image count: %s' % images.count())

    # TODO: how to do something like
    # result = model.predict(images.values())
    # print(result.collect())
    # ????

    for filename, image_data in images.collect():
        predict_result = bigdl_model.predict(image_data)
        print(predict_result)
        LOG.info('%s: %s' % (filename, predict_result[0].argmax()))

    sc.stop()


if __name__ == '__main__':
    main()
