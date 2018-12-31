import os, sys
import pandas as pd
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)


class UtilTools:
    def __init__(self):
        self.train_image_folder = os.environ.get('TRAIN_IMAGES')
        self.validation_image_folder = os.environ.get('VALIDATION_IMAGES')
        self.cropped_image_size = os.environ.get('FIXED_IMAGE_SIZE')

        self.logpath = os.environ.get('LOG_FOLDER')

        self.labels = 'label.scv'
        self.pixel_dict_pkl = 'pixel_dict.pkl'
        self.synset_map = 'int_to_string_labels.pbtxt'

    def createLogFile(self, filename):
        return self.logpath + filename

    def map_labels(self):
        label_string = []
        label_int = []
        with open(self.synset_map, 'r') as f:
            for line in f:
                for word in line.split():
                    if word == 'target_class_string:':
                        label_string.append(line.split()[1].replace('"', ''))
                    elif word == 'target_class:':
                        label_int.append(line.split()[1])

        df = pd.DataFrame({'IntLabel': label_int, 'StringLabel': label_string})
        print(df.head())

        df.to_csv(self.labels, index=False)
