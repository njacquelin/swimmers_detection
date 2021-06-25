import pickle
import sys

import pandas as pd
import os
import cv2
import numpy as np

import torch


def save_obj(obj, name) :
    name += '.jpg'
    cv2.imwrite(name, obj)


def normalize(coefs) :
    total = max(coefs)
    coefs = [c/total for c in coefs]
    return coefs


def distance(x1, y1, x2, y2, ratio=1) :
    dist = ((x1 * ratio - x2 * ratio) ** 2 + (y1 - y2) ** 2) ** 0.5
    return dist


def get_tensor(frame_data, path_img) :
    # img_size = frame_data['img_height'].iloc[0], frame_data['img_width'].iloc[0]
    img = cv2.imread(path_img)
    height, width, channels = img.shape
    img_size = height, width
    tensor = np.zeros(img_size)
    threshold = 1/3

    for (xmin, ymin, xmax, ymax, center_x, center_y) in \
            zip(frame_data['xmin'], frame_data['ymin'],
                frame_data['xmax'], frame_data['ymax'],
                frame_data['x'], frame_data['y']) :
        ratio = (ymax-ymin) / (xmax-xmin)

        # if ratio < threshold :
        #     half_delta_x = (xmax - xmin) / 2
        #     min_half_delta_y = int(half_delta_x * threshold)
        #     ymin = center_y - min_half_delta_y
        #     ymax = center_y + min_half_delta_y
        #     ratio = threshold
        #     # print('bigger potatoe')
        #     ymin = max(0, ymin)
        #     ymax = min(img_size[0], ymax)

        # ellipse size
        dist_max = max(ymax-ymin, xmax-xmin) / 2 * ratio

        # ellipse generator
        # for y in range(ymin, ymax) :
        #     for x in range(xmin, xmax):
        #         dist = distance(x, y, center_x, center_y, ratio)
        #         norm_dist = dist / dist_max
        #         inv_dist = 1 - norm_dist
        #         # tensor[y, x] = max(inv_dist * 255, tensor[y, x])
        #         if inv_dist > 0.1 :
        #             tensor[y, x] = max(255, tensor[y, x])

        # hard box
        # tensor[ymin:ymax, xmin:xmax] = 255

        # soft box
        for y in range(ymin, ymax) :
            for x in range(xmin, xmax):
                dist = distance(x, y, center_x, center_y, ratio) // 2
                norm_dist = dist / dist_max
                inv_dist = 1 - norm_dist ** 0.5
                tensor[y, x] = max(inv_dist * 255, tensor[y, x])
                # if inv_dist > 0.1 :
                #     tensor[y, x] = max(inv_dist * 255, tensor[y, x])

    return tensor




def get_and_adapt_general_data(path):
    data = pickle.load(open(path, 'rb'))
    data.sort_values(['frame'], inplace=True)

    # data['x'] = (data['xmin'] +  data['xmax']) // 2
    # data['y'] = (data['ymin'] + data['ymax']) // 2

    center_x = data['x']
    center_y = data['y']
    w = data['w']
    h = data['h']
    #
    data['xmin'] = center_x - w // 2
    data['ymin'] = center_y - h // 2
    data['xmax'] = center_x + w // 2
    data['ymax'] = center_y + h // 2
    return data

# def get_size_image(data, path_img):
#     img = cv2.imread('foo.jpg')
#     height, width, channels = img.shape
#     data['img_height'] = da
#     return data

if __name__ == '__main__':
    path = '/home/nicolas/swimmers_tracking/extractions/labels_pickle/dataframe_bboxes.pkl'

    img_repo = '/home/nicolas/swimmers_tracking/extractions/labelled_images/both/'

    positives_path = '/home/nicolas/unsupervised-detection/dataset/general/yes_softBox/'
    # importance_path = '/home/amigo/Bureau/data/video_for_extracting/importance'

    if not os.path.isdir(positives_path) : os.mkdir(positives_path)
    # if not os.path.isdir(importance_path): os.mkdir(importance_path)

    data = get_and_adapt_general_data(path)
    # data['img_height'] = data['frame'] # wut ?

    for i, f in enumerate(data['frame'].unique()) :
        img_path = os.path.join(img_repo, f)
        tensor_path = os.path.join(positives_path, f[:-4]) # [:-4] to remove '.jpg' 'coz added in save_obj
        # importance_tensor_path = os.path.join(importance_path, f[:-4])  # [:-4] to remove '.jpg'

        if i%50==0 :
            print(f)
        frame_data = data[data['frame'] == f]

        tensor = get_tensor(frame_data, img_path)

        save_obj(tensor, tensor_path)


