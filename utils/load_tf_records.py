from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def decode(serialized_example):
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                       })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    return image, label


def load(mode):
    dataset = tf.data.TFRecordDataset(mode+'.tfrecords')
    dataset = dataset.map(decode)
    # dataset = dataset.map(augment)
    # dataset = dataset.map(normalize)
    print(dataset)


if __name__ == '__main__':
    load('train')