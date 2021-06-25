import torch
from torch import save, load
from torch.nn import MSELoss, BCELoss
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from dataloader import get_train_test_dataloaders
from model import Unet_like, deeper_Unet_like
import os

from blobs_utils import get_last_batch_mAP
from random import random


if __name__=='__main__' :
    torch.cuda.empty_cache()
    # writer = SummaryWriter('runs/training')

    train_img_path =    '/home/nicolas/swimmers_tracking/extractions/labelled_images/train'
    test_img_path =     '/home/nicolas/swimmers_tracking/extractions/labelled_images/test'
    # out_path =          '/home/nicolas/unsupervised-detection/dataset/general/yes_hardEdges'
    # test_out_path =     '/home/nicolas/unsupervised-detection/dataset/general/yes_hardEdges'
    out_path =          '/home/nicolas/unsupervised-detection/dataset/general/yes_hardBox/'
    test_out_path =     '/home/nicolas/unsupervised-detection/dataset/general/yes_hardBox'
    # model = Unet_like().cuda()
    model = deeper_Unet_like().cuda()
    batch_size = 16
    models_path = './hardBox/'
    model_prefix = ''
    epochs_already_trained = 0

    # train_img_path = '/home/nicolas/swimmers_tracking/extractions/unlabelled and train images'
    # test_img_path = '/home/nicolas/swimmers_tracking/extractions/labelled_images/test'
    # out_path = '../dataset/general/pseudo_yes and yes'
    # test_out_path = '../dataset/general/yes_smooth'
    # model = Unet_like().cuda()
    # batch_size = 70
    # models_path = './student1_models/'
    # model_prefix = '/student1_'
    # epochs_already_trained = 115

    size = (256, 256)
    lr = 1e-3
    epochs_nb = 100
    optimizer_function = Adam
    save_after_N_epochs = 5

    train_dataloader = get_train_test_dataloaders(train_img_path, out_path, size, batch_size=batch_size,
                                                  train_test_ratio=1)
    test_dataloader = get_train_test_dataloaders(test_img_path, test_out_path, size, batch_size=batch_size,
                                                 train_test_ratio=1, augment_data=False)
    print('dataloader and model loaded')

    optimizer = optimizer_function(model.parameters(),
                                   lr=lr,
                                   weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, min_lr=1e-6, verbose=True)
    criterion = MSELoss()
    # criterion = ponderated_MSE
    # criterion = BCELoss()

    if not os.path.isdir(models_path) : os.mkdir(models_path)

    if epochs_already_trained != 0:
        model.load_state_dict(load(models_path + model_prefix + str(epochs_already_trained) + 'epochs.pth'))

    plt.ion()
    plt.show()

    all_losses = []
    all_training_losses = []
    all_mAP50 = []

    for epoch in range(epochs_already_trained, epochs_already_trained + epochs_nb) :

        ### TEST PART ###
        total_epoch_loss = 0
        model.train()
        for batch in train_dataloader :
            img = batch['img'].cuda()
            truth = batch['out'].cuda()

            out = model.forward(img)
            loss = criterion(out, truth)

            total_epoch_loss += float(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_epoch_loss /= len(train_dataloader)
        # writer.add_scalar('Loss', total_epoch_loss, epoch + 1)
        print('train :', total_epoch_loss, epoch+1)
        all_training_losses.append(total_epoch_loss)

        ### TRAIN PART ###
        model.eval()
        total_epoch_loss = 0
        with torch.no_grad() :
            for batch in test_dataloader :
                img = batch['img'].cuda()
                truth = batch['out'].cuda()
                out = model.forward(img)
                loss = criterion(out, truth)
                total_epoch_loss += float(loss)
            last_batch_mAP50 = get_last_batch_mAP(truth, out, 0.5, heatmap_threshold=0.5)
            all_mAP50.append(last_batch_mAP50)
            total_epoch_loss /= len(test_dataloader)
            all_losses.append(total_epoch_loss)

            # writer.add_scalar('Loss', total_epoch_loss, epoch + 1)

            plt.subplot(2, 1, 1)
            plt.plot(all_mAP50, '-+', label='mAP')
            plt.legend()
            plt.grid()

            plt.subplot(2, 1, 2)
            plt.plot(all_losses, '-o', label='test')
            plt.plot(all_training_losses, '-x', label='train')
            plt.legend()
            plt.grid()

            plt.show()
            plt.pause(0.1)
            print('test :', total_epoch_loss, epoch + 1
                  # , "\tentropy =", str(round(float(criterion(out, out)), 3)))
                  , "\tsample mAP 50 =", last_batch_mAP50)
            print()
        torch.cuda.empty_cache()

        scheduler.step(total_epoch_loss)

        if epoch % save_after_N_epochs == save_after_N_epochs - 1 :
            save(model.state_dict(), models_path + model_prefix + str(epoch + 1) + 'epochs.pth')
            print('Saved at epoch ' + str(epoch+1))