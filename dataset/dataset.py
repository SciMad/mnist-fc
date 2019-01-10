'''to work on the dataset generator'''

import numpy as np
import os, sys, glob
import tensorflow as tf
import pickle
from configs.config import config

class DataGenerator:
    def __init__(self, config):
        self.config = config
        # get mode, 'train' and 'val' and set the self.mode or use the config?

        data_file = open(self.config['mnist-data'], 'rb')
        u = pickle._Unpickler(data_file)
        u.encoding = 'latin1'
        self.train_data, self.validate_data, self.test_data = u.load()

    def train_gen(self):
        i = 0
        while True:
            yield (self.train_data[0][i], self.train_data[1][i])
            i += 1


    def load(self, mode):
        print("Trying to Create Iterator for ", mode, "mode")
        if mode == 'train':
            dataset = tf.data.Dataset.from_generator(self.train_gen,
                                                     (tf.float32, tf.int64))
            dataset = dataset.shuffle(buffer_size=50000)
        elif mode == 'test':
            dataset = tf.data.Dataset.from_generator(self.test_gen,
                                                     (tf.float32, tf.int64))
            dataset = dataset.shuffle(buffer_size=10000)

        #dataset = dataset.batch(config['batch_size'])

        print("dataset:", dataset)
        iterator = dataset.make_initializable_iterator()
        print ("Returning Iterator")
        return iterator

if __name__ == '__main__':
    print("Verifying Dataset")
    data_gen = DataGenerator(config)
    iterator = data_gen.load('train')

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        next_element = iterator.get_next()
        while (True):
            try:
                data_point = sess.run(next_element)
                print(data_point)
            except Exception as e:
                print(e)
                break
