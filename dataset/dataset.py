'''to work on the dataset generator'''

import numpy as np
import os, sys, glob


class DataGenerator:
    def __init__(self, config):
        self.config = config
        # get mode, 'train' and 'val' and set the self.mode or use the config? 

    def list_tf_records(self, datadir):
    	'''A helper function to list the tfrecords file in the data dir.'''
    	return glob.glob(datadir + "*.tfrecords")

    def decode(serialized_example):
	    features = tf.parse_single_example(serialized_example,
	                                       features={
	                                           'image_raw': tf.FixedLenFeature([], tf.string),
	                                           'label': tf.FixedLenFeature([], tf.int64),
	                                       })
	    image = tf.decode_raw(features['image_raw'], tf.uint8)
	    label = tf.cast(features['label'], tf.int32)
	    #one_hot = tf.one_hot(label, NUM_CLASSES)
	    return image, label


	def load(self, mode):
		
		filenames = tf.placeholder(tf.string, shape=[None])
		dataset = tf.data.TFRecordDataset(filenames)
		dataset = dataset.map(decode)
		dataset = dataset.shuffle(buffer_size=10000)
		dataset = dataset.batch(32)
		iterator = dataset.make_initializable_iterator()

		# You can feed the initializer with the appropriate filenames for the current
		# phase of execution, e.g. training vs. validation.

		# Initialize `iterator` with training data.
		
		training_filenames = self.list_tf_records(config['traindatadir'])
		
		#way to load the dataset.
		#sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

		# Initialize `iterator` with validation data.
		validation_filenames = self.list_tf_records(config['valdatadir'])

		return iterator, training_filenames, validation_filenames
		