import numpy as np
import os, sys
from PIL import Image
import pickle
from .util import UtilTools

tool = UtilTools()
train_image_folder = tool.train_image_folder
validation_image_folder = tool.validation_image_folder
cropped_image_size = tool.cropped_image_size


def resize_image(image, h, w):
    aspect_ratio = w/h
    if aspect_ratio == 1 or h < cropped_image_size or w < cropped_image_size:
        return image.resize((cropped_image_size, cropped_image_size))
    if aspect_ratio < 1:
        ratio = cropped_image_size/w
        image = image.resize((cropped_image_size, int(h*ratio)), Image.ANTIALIAS)
    elif aspect_ratio > 1:
        ratio = cropped_image_size/h
        image = image.resize((int(w*ratio), cropped_image_size), Image.ANTIALIAS)
    height, width = image.size
    left = (width - cropped_image_size) / 2
    top = (height - cropped_image_size) / 2
    right = (width + cropped_image_size) / 2
    bottom = (height + cropped_image_size) / 2

    return image.crop((left, top, right, bottom))


def get_mean(image):
    pixels = np.asarray(image, dtype="int32")
    rgb = pixels[0]
    mean = np.mean(rgb, axis=0)
    # print(mean)
    return mean, pixels


def save_as_pickle(data, filename):
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filename):
    with open(filename + '.pkl', 'rb') as f:
        return pickle.load(f)


def pre_process():
    mean_dict = {}
    pixel_dict = {}
    for path, sub_dir, files in os.walk(train_image_folder):
        for name in files:
            current_image = os.path.join(path, name)
            image = Image.open(current_image)
            height, width = image.size
            cropped_image = resize_image(image,height,width)
            # cropped_image.save('transformed/'+ current_image.split('/')[2])
            mean, pixels = get_mean(cropped_image)
            image_id = current_image.split('/')[2].split('.')[0]
            mean_dict[image_id] = mean
            pixel_dict[image_id] = pixels

    reduced_mean = np.mean(list(mean_dict.values()), axis=0)
    for key, items in pixel_dict.items():
        rgb = items[0] - reduced_mean
        pixel_dict[key][0] = rgb
    print(reduced_mean)
    print(pixel_dict.values())
    save_as_pickle(pixel_dict, tool.pixel_dict_pkl)


if __name__ == '__main__':
    pre_process()
