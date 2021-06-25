import torch
from torch import load, unsqueeze, stack, no_grad
from torchvision import transforms

import os
from skimage import io, img_as_ubyte
from skimage.transform import resize
from cv2 import addWeighted
import cv2
import numpy as np
from matplotlib import pyplot as plt

from model import Unet_like, fatass_Unet_like


def compare(out, img, thresholod=None):
    heatmap = np.absolute(out - img)
    if thresholod is not None :
        heatmap = np.where(heatmap > thresholod, 1., 0.)
    heatmap = np.amax(heatmap, 2)
    heatmap = np.stack([heatmap, heatmap, heatmap], axis=2)
    return heatmap


def tensor_to_image(out, inv_trans=True) :
    std = torch.tensor([0.229, 0.224, 0.225])
    mean = torch.tensor([0.485, 0.456, 0.406])
    if inv_trans :
        for t, m, s in zip(out, mean, std):
            t.mul_(s).add_(m)
    out = out.cpu().numpy()
    out *= 255
    out = out.astype(np.uint8)
    # out = out.astype(np.float64)
    out = np.swapaxes(out, 0, 2)
    out = np.swapaxes(out, 0, 1)
    return out


def get_transform(x) :
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])
    tensor = img_transform(x)
    tensor = unsqueeze(tensor, 0).float()
    return tensor.cuda()


def get_video_name(epochs, full_images_path) :
    video_frames = full_images_path.split('/')[-1]
    video = video_frames[:-7]
    video_epochs = video + "_" + str(epochs)
    video_epochs_avi = 'heatmap_' + video_epochs + '.avi'
    return video_epochs_avi


def init_blob_detector() :
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 127
    params.maxThreshold = 129
    params.thresholdStep = 1
    params.filterByArea = True
    params.minArea = 10
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.minDistBetweenBlobs = 1
    detector = cv2.SimpleBlobDetector_create(params)
    # detector = cv2.SimpleBlobDetector_create()
    return detector


def extract_blobs(out, detector) :
    out = 1 - out
    out = cv2.threshold(out, 127, 255, cv2.THRESH_BINARY)[1]
    # keypoints = detector.detect(out)
    keypoints = detector.findBlobs(out)
    if len(keypoints) != 0 :
        pass

    out2 = np.expand_dims(out, 2)
    heatmap = np.concatenate((out2, out2, out2), axis=2)
    for p in keypoints :
        x = int(p.pt[0])+1
        y = int(p.pt[1])+1
        offset = int(p.size / 2)+1
        heatmap[y, x - offset : x + offset] = (0, 255, 0)
        heatmap[y - offset : y + offset, x] = (0, 255, 0)
    # heatmap = cv2.drawKeypoints(heatmap, keypoints, np.array([]), (255),
    #                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return heatmap


if __name__=='__main__':
    epochs = 55

    size = (512, 512)

    path = 'yes_' + str(epochs) + 'epochs.pth'
    models_path = './models'

    # full_images_path = '../../PhD_HPE/data/images/2_blackmagic_videos/high_rez_breaststroke'
    # full_images_path = '../../PhD_HPE/data/images/2_blackmagic_videos/high_rez_crawl'
    # full_images_path = '/home/nicolas/swimmers_tracking/extractions/Gwangju_frames'
    # full_images_path = '/home/nicolas/swimmers_tracking/extractions/Angers19_frames'
    full_images_path = '/home/nicolas/swimmers_tracking/extractions/Rennes19_frames'
    # full_images_path = '/home/nicolas/swimmers_tracking/extractions/TITENIS_frames'

    video_name = get_video_name(epochs, full_images_path)
    video_path_heatmap = './videos/' + video_name

    model = Unet_like()
    # model = fatass_Unet_like()
    model_path = os.path.join(models_path, path)
    model.load_state_dict(load(model_path))
    model = model.cuda()
    model.eval()

    video_flow_heatmap = cv2.VideoWriter(video_path_heatmap, cv2.VideoWriter_fourcc(*'XVID'), 25, (size[1], size[0]))

    detector = init_blob_detector()

    for root, dirs, files in os.walk(full_images_path) :
        files.sort()
        files.sort(key=len, reverse=False)

        for i, file in enumerate(files) :
            img_path = os.path.join(root, file)
            img_source = io.imread(img_path)
            img = resize(img_source, size)
            tensor_img = get_transform(img)
            with no_grad() :
                out = model(tensor_img)[0]
            out = tensor_to_image(out, False)
            heatmap = extract_blobs(out, detector)
            video_flow_heatmap.write(img_as_ubyte(heatmap))
            print(i)
        video_flow_heatmap.release()