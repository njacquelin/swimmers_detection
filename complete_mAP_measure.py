import torch
from torch import save, load

import matplotlib.pyplot as plt
from dataloader import get_train_test_dataloaders
from model import deeper_Unet_like, Unet_like
import os

from blobs_utils import get_last_batch_mAP, get_last_batch_mAR, get_last_batch_mAPR


if __name__=='__main__' :
    torch.cuda.empty_cache()

    test_img_path = '/home/nicolas/swimmers_tracking/extractions/labelled_images/test'
    test_out_path = '/home/nicolas/unsupervised-detection/dataset/general/yes_hardEdges'
    batch_size = 64
    models_path = './softBlob/'
    # models_path = './models/'

    model_prefix = ''
    model = deeper_Unet_like().cuda()
    epochs_already_trained = 65
    # model_prefix = 'colorShifts_deeper_zoomedOut_'
    # model = deeper_Unet_like().cuda()
    # epochs_already_trained = 200
    # model_prefix = 'yes_'
    # model = Unet_like().cuda()
    # epochs_already_trained = 130

    # size = (128, 128)
    size = (256, 256)
    # size = (512, 512)

    test_dataloader = get_train_test_dataloaders(test_img_path, test_out_path, size, batch_size=batch_size,
                                                 train_test_ratio=1, augment_data=False)
    print('dataloader and model loaded')

    if not os.path.isdir(models_path) : os.mkdir(models_path)

    model.load_state_dict(load(models_path + model_prefix + str(epochs_already_trained) + 'epochs.pth'))
    model.eval()

    mAP = 0
    mAR = 0
    divisor = 0

    with torch.no_grad() :
        for batch in test_dataloader :
            img = batch['img'].cuda()
            truth = batch['out'].cuda()
            out = model.forward(img)

            IOU_threshold = 0.25
            batch_mAP, batch_mAR = get_last_batch_mAPR(truth, out, threshold=IOU_threshold, heatmap_threshold=0.45, min_blob_size=None)
            mAP += batch_mAP * img.shape[0]
            mAR += batch_mAR * img.shape[0]

            divisor += img.shape[0]
    mAP /= divisor
    mAR /= divisor
    print("mAP " + str(int(IOU_threshold*100)) + " =", mAP)
    print("mAR " + str(int(IOU_threshold*100)) + " =", mAR)
