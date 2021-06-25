import torch
from torch import load, unsqueeze, stack, no_grad
from torch.cuda import empty_cache
from torchvision import transforms

import os
from skimage import io
from skimage.transform import resize
from cv2 import addWeighted
import numpy as np
import cv2
from matplotlib import pyplot as plt
import gc

from model import Unet_like


def tensor_to_image(out) :
    out = out.cpu().numpy() * 255
    out = out.astype(np.uint8)
    out = np.swapaxes(out, 0, 2)
    out = np.swapaxes(out, 0, 1)
    # out *= 255
    return out


def get_transform(x) :
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])
    tensor = img_transform(x).float()
    tensor = unsqueeze(tensor, 0)
    return tensor.cuda()


def save_obj(obj, name) :
    name += '.jpg'
    cv2.imwrite(name, obj)


if __name__=='__main__':
    epochs = 55

    models_path = './models'
    path = 'yes_'+str(epochs)+'epochs.pth'
    model_path = os.path.join(models_path, path)

    full_images_path = '/home/nicolas/swimmers_tracking/extractions/unlabelled_images'
    out_path = '../dataset/general/pseudo_yes'

    size = (512, 512)

    model = Unet_like()
    model.load_state_dict(load(model_path))
    model = model.cuda()
    model.eval()

    for root, dirs, files in os.walk(full_images_path) :
        files.sort()
        files.sort(key=len, reverse=False)
        for i, file in enumerate(files) :
            img_path = os.path.join(root, file)
            img = io.imread(img_path)
            img = resize(img, size)
            tensor_img = get_transform(img)
            with no_grad() :
                out = model(tensor_img)[0]
            out = tensor_to_image(out)
            out = np.where(out < 200, 0, 255)
            out_file_path = os.path.join(out_path, file)
            cv2.imwrite(out_file_path, out)
