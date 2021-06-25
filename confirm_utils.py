import cv2
import torch
from torchvision import transforms


def get_subtensor(img, half_size, xmin, ymin, xmax, ymax) :
        x_center = (xmin + xmax) // 2
        y_center = (ymin + ymax) // 2

        if x_center < half_size[1] : x_center = half_size[1]
        if x_center > img.shape[2] - half_size[1]: x_center = img.shape[2] - half_size[1]
        if y_center < half_size[0] : y_center = half_size[0]
        if y_center > img.shape[1] - half_size[0] : y_center = img.shape[1] - half_size[0]

        subimg = img[:, y_center - half_size[0]:y_center + half_size[0],
                    x_center - half_size[1]:x_center + half_size[1]]
        return subimg


def get_subimg(img, half_size, xmin, ymin, xmax, ymax):
    x_center = (xmin + xmax) // 2
    y_center = (ymin + ymax) // 2

    if x_center < half_size[1]: x_center = half_size[1]
    if x_center > img.shape[2] - half_size[1]: x_center = img.shape[1] - half_size[1] - 1
    if y_center < half_size[0]: y_center = half_size[0]
    if y_center > img.shape[1] - half_size[0]: y_center = img.shape[0] - half_size[0] - 1

    subimg = img[y_center - half_size[0]:y_center + half_size[0],
             x_center - half_size[1]:x_center + half_size[1]]
    return subimg


def get_transform(img):
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])
    return img_transform(img)


from matplotlib import pyplot as plt
def confirm_boxes(boxes, confirm_model, cv_img, size = (128,128), confirm_threshold=0.9) :
    if boxes == [] : return []
    img = get_transform(cv_img)
    half_size = size[0]//2, size[1]//2
    tensor = torch.empty((len(boxes), 3, size[0], size[1]))
    for i, (xmin, ymin, xmax, ymax) in enumerate(boxes) :
        subtensor = get_subtensor(img, half_size, xmin, ymin, xmax, ymax)
        tensor[i] = subtensor
    tensor = tensor.cuda()
    out = confirm_model(tensor)
    accepted_boxes = []
    for result, box in zip(out, boxes) :
        if result > confirm_threshold :
            accepted_boxes.append(box)
            # xmin, ymin, xmax, ymax = box
            # subimg = get_subimg(cv_img, half_size, xmin, ymin, xmax, ymax)
            # plt.title("keep " + str(result))
            # plt.imshow(subimg)
            # plt.show()
    return accepted_boxes