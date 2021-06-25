from torchvision import transforms
from torch.autograd import Variable
import torch

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import cv2

from dataloader import get_train_test_dataloaders


def torch2np(img, inv_trans=True, float_to_uint8=True) :
    invTrans = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                   transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),
                                  ])
    if inv_trans : img = invTrans(img)
    img = img.permute(1, 2, 0)
    img = img.numpy()
    if float_to_uint8 :
        img *= 255
        img = img.astype(np.uint8)
    return img


def generate_expected_out(reverse_mask, expected_output, out) :
    return out*reverse_mask + expected_output


if __name__=='__main__' :
    # img_path = '/home/amigo/Bureau/data/video_for_extracting/renaud+nicolas_images'
    img_path = '/home/nicolas/swimmers_tracking/extractions/labelled_images/train'
    # out_path = '/home/amigo/Bureau/data/video_for_extracting/renaud+nicolas_heatmaps'
    out_path = '/home/nicolas/unsupervised-detection/dataset/general/yes_hardEdges'
    size = (256, 256)

    train_dataloader = get_train_test_dataloaders(img_path, out_path, size, batch_size=1, train_test_ratio=1, augment_data=False, shuffle=False)

    for batch in train_dataloader:
        image = batch['img'][0]
        img = torch2np(image)

        out = batch["out"][0]
        out = torch2np(out, inv_trans=False)
        out = np.concatenate((out, out, out), axis=2)

        # img = cv2.addWeighted(img, 0.5, out, 0.5, 0)

        plt.imshow(img)
        plt.show()