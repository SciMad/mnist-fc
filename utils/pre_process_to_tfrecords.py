from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os, sys
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
sys.path.append(os.environ.get('PROJECT_FOLDER'))
from PIL import Image
import pickle
import pandas as pd
from natsort import natsorted
from util import UtilTools
import time
import tensorflow as tf


class PreProcess:
    def __init__(self, image_folder, pixel_dict_folder, pixel_dict_filename, image_size):
        self.image_folder = image_folder
        self.pixel_dict_folder = pixel_dict_folder
        self.pixel_dict_filename = pixel_dict_filename
        self.image_size = image_size

        self.image_path_pickle = 'image_path.pth'
        self.mean_dict_filename = '_mean_dict.pkl'
        self.preprocessed_dict_pkl_filename = 'preprocessed.pkl'

        self.record_filename = 'record.tfrecords'
        self.pickle_image_path()

    def pickle_image_path(self):
        filename = self.pixel_dict_folder+self.image_path_pickle
        if not os.path.exists(filename):
            print('pickling image path...')
            image_path = []
            for path, sub_dir, files in os.walk(self.image_folder):
                for name in files:
                    image_path.append(os.path.join(path, name))
            save_as_pickle(image_path, self.pixel_dict_folder+self.image_path_pickle)
            print('Image path saved in file {}'.format(filename))

    def initialize_processing(self):
        start_time = time.time()
        count = 0

        print('We are in image {}'.format(count))
        mean_dict = {}
        print('Loading the pickled image path...')
        image_path = load_pickle(self.pixel_dict_folder+self.image_path_pickle)
        image_path = image_path[count:]
        for current_image in image_path:
            try:
                print(current_image, count)
                image = Image.open(current_image)
                height, width = image.size
                cropped_image = self.resize_image(image, height, width)
                _, mean = get_mean(cropped_image)
                image_id = current_image.split('/')[-1].split('.')[0]
                print(image_id)
                mean_dict[image_id] = mean

            except BaseException as e:
                print(e)

            count += 1
            if count % 20000 == 0:
            
                save_as_pickle(mean_dict, str(count)+'_mean_dict.pkl')

        print("--- %s seconds ---" % (time.time() - start_time))
        save_as_pickle(mean_dict, self.mean_dict_filename)
        print('Saved mean values of {} images successfully.'.format(count))

    def normalize_rgb(self):
        count = 0
        df = pd.read_csv('../data/label.csv').set_index('IntLabel')
        filename = os.path.join(self.pixel_dict_folder, self.record_filename)
        mean_dict = load_pickle(self.pixel_dict_folder+self.mean_dict_filename)
        reduced_mean = np.mean(list(mean_dict.values()))

        image_path = load_pickle(self.pixel_dict_folder + self.image_path_pickle)
        image_path = image_path[count:]
        for current_image in image_path[100]:
            try:
                print(current_image)
                image = Image.open(current_image)
                height, width = image.size
                cropped_image = self.resize_image(image, height, width)
                pixels = np.asarray(cropped_image.convert('RGB'), dtype="int32")
                pixels[0] = pixels[0] - reduced_mean
                image_id = current_image.split('/')[-1].split('.')[0]
                convert_to_examples(filename, df, image_id, pixels)

            except BaseException as e:
                print(e)

            count += 1

    def resize_image(self, image, h, w):
        """
        :param image: Image
        :param h: height
        :param w: width
        :return: Image cropped to size 256X256
        """
        aspect_ratio = w / h
        if aspect_ratio == 1 or h < self.image_size or w < self.image_size:
            return image.resize((self.image_size, self.image_size))
        if aspect_ratio < 1:
            ratio = self.image_size / w
            image = image.resize((self.image_size, int(h * ratio)), Image.ANTIALIAS)
        elif aspect_ratio > 1:
            ratio = self.image_size / h
            image = image.resize((int(w * ratio), self.image_size), Image.ANTIALIAS)
        height, width = image.size
        left = (width - self.image_size) / 2
        top = (height - self.image_size) / 2
        right = (width + self.image_size) / 2
        bottom = (height + self.image_size) / 2
        return image.crop((left, top, right, bottom))


def get_labels(row_label, df):
    """
    :param row_label: the synset id
    :param df: data_frame where the mapping of synset to the integer label is stored
    :return: corresponding integer id
    """
    a = df.index[df['StringLabel'] == row_label.split('_')[0]]
    print(a[0])
    return a[0]


def _int64_feature(value):
    """
    :param value: value to be converted to int64
    :return: converted value
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """
    :param value: value to be converted into bytes
    :return: converted value
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_examples(filename, df, imageid, value):
    with tf.python_io.TFRecordWriter(filename) as writer:
        image_raw = value.tostring()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'height': _int64_feature(value.shape[1]),
                    'width': _int64_feature(value.shape[2]),
                    'depth': _int64_feature(value.shape[3]),
                    'label': _int64_feature(int(get_labels(imageid, df))),
                    'image_raw': _bytes_feature(image_raw)
                }))
        writer.write(example.SerializeToString())


def save_as_pickle(data, filename):
    """
    :param data: data to save as pickle
    :param filename: filename in which to save the image
    """
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print('Data saved in {}'.format(filename))


def get_mean(image):
    """
    :param image: cropped image of shape: 256X256X3
    :return: mean: Mean rgb of the input image of shape: 1X3
    :return: pixels: array of the image
    """
    pixels = np.asarray(image.convert('RGB'), dtype="int32")
    rgb = pixels[0]
    mean = np.mean(rgb, axis=0)
    return pixels, mean


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def run_pre_processing(command):
    tool = UtilTools()
    image_size = tool.cropped_image_size
    if command == 'train':
        image_folder = tool.train_image_folder
        pixel_dict_folder = tool.train_data

    elif command == 'eval':
        image_folder = tool.validation_image_folder
        pixel_dict_folder = tool.validation_data
    else:
        print('Wrong command')
        sys.exit(1)

    pixel_dict_filename = tool.dict_pickle

    pre_process = PreProcess(image_folder, pixel_dict_folder, pixel_dict_filename, image_size)
    #pre_process.initialize_processing()
    pre_process.normalize_rgb()

if __name__=='__main__':
    # available modes are: start and last
    run_pre_processing('train')