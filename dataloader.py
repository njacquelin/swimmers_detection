import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import os
from skimage import io
import cv2
from cv2 import resize, GaussianBlur, findHomography, warpPerspective
import numpy as np
from random import random


class Yes_Dataloader(Dataset) :
    def __init__(self, img_path, out_path, size, augment_data=True) :
        self.transform = self.get_transform()
        self.augment_data = augment_data
        self.size = size

        yes_img = os.listdir(img_path)
        yes_img = [os.path.join(img_path, f) for f in yes_img]
        yes_img.sort()
        self.yes_img = yes_img

        yes_out = os.listdir(out_path)
        dico = {}
        for f in yes_out :
            dico[f] = os.path.join(out_path, f)
        # yes_out = [os.path.join(out_path, f) for f in yes_out]
        # yes_out.sort()
        self.yes_out = dico # yes_out

        self.len = len(self.yes_img)

    def __len__(self) :
        return self.len

    def __getitem__(self, idx):
        img_path = self.yes_img[idx]
        img = io.imread(img_path)

        img_name = img_path.split('/')[-1]
        out_name = self.yes_out[img_name]
        out = io.imread(out_name)

        big_augment = False

        if self.augment_data and random() > 0.5:  # gaussian blurr
            kernel = 3 if random() > 0.5 else 5 if random() > 0.5 else 7 if random() > 0.5 else 9
            img = GaussianBlur(img, (kernel, kernel), 0)

        if self.augment_data and random() > 0.5:  # crop
            if random() > 0.5 :
                img, out = self.random_crop(img, out)
            else :
                img, out = self.zoom_out(img, out)
            big_augment = True

        if self.augment_data and random() > 0.7 :
            img = self.color_shift(img)

        # if self.augment_data:  # and not big_augment and random() > 0.5: # homography
        #     if random() > 0.5:
        #         img, out = self.homography_augmentation(img, out) # NOTE : cancelled => returns img, out directly
        #     else:
        #         img, out = self.stretch_augmentation(img, out)
        #     big_augment = True

        img = resize(img, self.size)
        out = resize(out, self.size)
        img = self.transform(img)
        out = self.to_tensor(out)

        if self.augment_data :
            img, out = self.flips(img, out)
        # if random() > 0.9 and not big_augment : # left_right_switch but not if already crop
        #     img, out = self.left_right_switch(img, out)

        return {'img': img, 'out': out}

    def zoom_out(self, img, out) :
        reducing_factor = 2 + random() * 3 # [2, 5[
        # reducing_factor = 1 + random() * 2 # [1, 3[
        new_size = int(img.shape[1] / reducing_factor), int(img.shape[0] / reducing_factor)
        x_margin = int(random() * (img.shape[0] - new_size[1]))
        y_margin = int(random() * (img.shape[1] - new_size[0]))

        grey = np.ones_like(img) * 127
        img = cv2.resize(img, new_size)
        grey[x_margin : x_margin + new_size[1],
             y_margin : y_margin + new_size[0]] = img

        black = np.zeros_like(out)
        out = cv2.resize(out, new_size)
        black[x_margin : x_margin + new_size[1],
              y_margin : y_margin + new_size[0]] = out

        return grey, black


    def stretch_augmentation(self, img, out):
        Y_size = int(self.size[0] * ((random() + 1) * 4))
        pos = int(random() * (Y_size - self.size[0]))
        img = resize(img, (self.size[0], Y_size))
        img = img[pos:pos + self.size[0]]
        out = resize(out, (self.size[0], Y_size))
        out = out[pos:pos + self.size[0]]
        return img, out

    def homography_augmentation(self, img, out):
        return img, out
        # pts_src, pts_dst = [], []
        # y_shift_max = 150 # img.shape[0] / 50
        # x_shift_max = 150 # img.shape[1] / 50
        # for line in range(2) :
        #     for col in range(2):
        #         y_src = int(random() * img.shape[0] // 4) + 3 * col * (img.shape[0] // 4)
        #         y_dst = int(y_src + 2 * (random() * y_shift_max) - random() * y_shift_max)
        #         x_src = int(random() * img.shape[1] // 4) + 3 * line * (img.shape[1] // 4)
        #         x_dst = int(x_src + 2 * (random() * x_shift_max) - random() * x_shift_max)
        #         pts_src.append([y_src, x_src])
        #         pts_dst.append([y_dst, x_dst])
        #
        # # test stuff, don't remove please future Nico...
        # # pts_src = [[50, 50], [150, 150], [150, 50], [50, 150]]
        # # pts_dst = [[50, 50], [150, 150], [150, 50], [50, 150]]
        #
        # pts_dst = np.array(pts_dst)
        # pts_src = np.array(pts_src)
        # h, _ = findHomography(pts_src, pts_dst)
        # # if h[0,0] < 0 : h[0,0] = -h[0,0] # avoids weird warps
        #
        # (heigh, width) = (img.shape[0], img.shape[1])
        # img = warpPerspective(img, h, (int(1*width), int(1*heigh)))
        # out = warpPerspective(out, h, (int(1*width), int(1*heigh)))
        # return img, out

    def get_transform(self):
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])
        return img_transform

    def random_crop(self, img, out) :
        min = 0.2 # side of the cropped image in respect to the original one
        prop = random() * (1 - min) + min

        h, w, _ = img.shape
        xmin = int(random() * w * (1 - prop))
        xmax = int(xmin + prop * w)
        ymin = int(random() * h * (1 - prop))
        ymax = int(ymin + prop * h)

        h_out, w_out = out.shape
        xmin_out = int(xmin * w_out / w)
        xmax_out = int(xmax * w_out / w)
        ymin_out = int(ymin * h_out / h)
        ymax_out = int(ymax * h_out / h)

        img = img[ymin:ymax, xmin:xmax]
        out = out[ymin_out:ymax_out, xmin_out:xmax_out]

        return img, out


    def color_shift(self, img) :
        brightness = int(random() * 128 - 64)
        contrast = int(random() * 128 - 64)
        hue = random() * 40 - 20

        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        img = cv2.addWeighted(img, alpha_b, img, 0, gamma_b)

        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        img = cv2.addWeighted(img, alpha_c, img, 0, gamma_c)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img)
        hnew = np.mod(h + hue, 180).astype(np.uint8)
        hsv = cv2.merge([hnew, s, v])
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return img

    def flips(self, img, x_out) :
        if random() > 0.5 :  # left-right inversion
            img = torch.flip(img, dims=[2])
            x_out = torch.flip(x_out, dims=[2])

        if random() > 0.5: # 90° rotation
            img = img.transpose(1, 2)
            x_out = x_out.transpose(1, 2)

        return img, x_out

    def left_right_switch(self, img, x_out) :
        tensors = [img, x_out]
        d_t, h_t, w_t = img.size()
        if random() > 0.5:  # left/right
            if random() > 0.5:  # lateral shift
                for tensor in tensors :
                    a = tensor[:, :h_t // 2].clone()
                    tensor[:, :h_t // 2] = tensor[:, h_t // 2:]
                    tensor[:, h_t // 2:] = a

            if random() > 0.5 : # grey on the top
                img[:, :h_t // 2] = torch.zeros_like(img[:, :h_t // 2]) * 0.5
                x_out[:, :h_t // 2] = torch.zeros_like(x_out[:, :h_t // 2])
            else : # grey at the bottom
                img[:, h_t // 2:] = torch.zeros_like(img[:, h_t // 2:]) * 0.5
                x_out[:, h_t // 2:] = torch.zeros_like(x_out[:, h_t // 2:])

        else:  # top/bottom
            if random() > 0.5:  # vertical shift
                for tensor in tensors :
                    a = tensor[:, :, :w_t // 2].clone()
                    tensor[:, :, :w_t // 2] = tensor[:, :, w_t // 2:]
                    tensor[:, :, w_t // 2:] = a

            if random() > 0.5 : # grey on the left
                img[:, :, :w_t // 2] = torch.zeros_like(img[:, :, :w_t // 2]) * 0.5
                x_out[:, :, :w_t // 2] = torch.zeros_like(x_out[:, :, :w_t // 2])
            else : # grey on the right
                img[:, :, w_t // 2:] = torch.zeros_like(img[:, :, w_t // 2:]) * 0.5
                x_out[:, :, w_t // 2:] = torch.zeros_like(x_out[:, :, w_t // 2:])

        return img, x_out

    def to_tensor(self, x):
        x = transforms.ToTensor()(x)
        return x


def get_train_test_dataloaders(img_path, out_path, size, batch_size=32, train_test_ratio=0.8, augment_data=True, shuffle=True):
    dataset = Yes_Dataloader(img_path, out_path, size, augment_data)
    if train_test_ratio != 1 :
        train_size = int(train_test_ratio * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_dataset.data_augmentation = False
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        return train_dataloader, test_dataloader
    else :
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader