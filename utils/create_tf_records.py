from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .util import UtilTools

import argparse
import os
import sys
import numpy as np

import tensorflow as tf
import pickle

import pandas as pd

FLAGS = None

tool = UtilTools()
train_image_folder = tool.train_image_folder
validation_image_folder = tool.validation_image_folder
cropped_image_size = tool.cropped_image_size


def load_pickle(filename):
    with open(filename + '.pkl', 'rb') as f:
        return pickle.load(f)


def get_labels(row_label, df):
    a = df.index[df['StringLabel'] == row_label.split('_')[0]]
    print(a[0])
    return a[0]


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name):
  """Converts a dataset to tfrecords."""
  images = np.array(list(data_set.values()))
  labels = np.array(list(data_set.keys()))
  num_examples = len(labels)
  print('There are {} examples', num_examples)

  if images.shape[0] != num_examples:
    raise ValueError('Images size %d does not match label size %d.' %
                     (images.shape[0], num_examples))
  rows = images.shape[1]
  cols = images.shape[2]
  depth = images.shape[3]

  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  print('Writing', filename)
  df = pd.read_csv(tool.labels).set_index('IntLabel')
  with tf.python_io.TFRecordWriter(filename) as writer:
    for index in range(num_examples):
      image_raw = images[index].tostring()
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'height': _int64_feature(rows),
                  'width': _int64_feature(cols),
                  'depth': _int64_feature(depth),
                  'label': _int64_feature(int(get_labels(labels[index], df))),
                  'image_raw': _bytes_feature(image_raw)
              }))
      writer.write(example.SerializeToString())


def main(unused_argv):
  # Get the data.
  data_sets = load_pickle(tool.pixel_dict_pkl)

  # Convert to Examples and write the result to TFRecords.
  convert_to(data_sets, 'train')
  # convert_to(data_sets.validation, 'validation')
  # convert_to(data_sets.test, 'test')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--directory',
      type=str,
      default='data/',
      help='Directory to download data files and write the converted result'
  )
  parser.add_argument(
      '--validation_size',
      type=int,
      default=5000,
      help="""\
        Number of examples to separate from the training data for the validation
        set.\
        """
  )

  FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)