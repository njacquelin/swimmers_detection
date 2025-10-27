# Description

This repo corresponds to the implementation of [this paper](https://hal.science/hal-03358375) titled : Detecting Swimmers in Unconstrained Videos with Few Training Data.

Please, refer to it by citing:

    @inproceedings{jacquelin:hal-03358375,
      TITLE = {{Detecting Swimmers in Unconstrained Videos with Few Training Data}},
      AUTHOR = {Jacquelin, Nicolas and Vuillemot, Romain and Duffner, Stefan},
      URL = {https://hal.science/hal-03358375},
      BOOKTITLE = {{Machine Learning and Data Mining for Sports Analytics}},
      ADDRESS = {Ghand, Belgium},
      YEAR = {2021},
      MONTH = Sep,
      KEYWORDS = {Computer Vision ; Swimming ; Small Dataset ; Segmentation},
      PDF = {https://hal.science/hal-03358375v1/file/Detecting%20Swimmers%20in%20Unconstrained%20Videos%20with%20Few%20Training%20Data.pdf},
      HAL_ID = {hal-03358375},
      HAL_VERSION = {v1},
    }



# Setup

Install the requirements. If some are missing, don't hesitate to add them to the list.  
The project has originally been done in python 3.8


# Models Uses

Models weights present in ./modesl directory.  
colorShifts_deeper_zoomedOut_200epochs.pth needs deeper_Unet_like model  
less_dataAug_130epochs.pth needs Unet_like model  
Anaylse a video with file video_display.py


# Training

For the training you need to have images of swimmers and there heatmap where the heatmap identifies the position of the swimmers.
So you need to create the heatmaps corresponding to your bounding boxes. The file "data_gen.py" may help or the 

Let's go in directory yes  
In the "trainer.py" file, you face this :

First choose the path to the data: train and test images and heatmaps:
```python
train_img_path = '/home/nicolas/swimmers_tracking/extractions/labelled_images/train' # the directory containing your training images  
test_img_path = '/home/nicolas/swimmers_tracking/extractions/labelled_images/test' # the directory containing your test images  
out_path = '../dataset/general/yes_smooth' # the directory containing your training data (heatmaps with the name of the original image)  
test_out_path = '../dataset/general/yes_smooth' # the directory containing your training data (heatmaps with the name of the original image)
```
choose the model. A deeper model will take more time to train.
Comment one of those two line to use the other model. Currently, the smaller model is used in the pipeline. 
```python
model = Unet_like().cuda()  
model = deeper_Unet_like().cuda() # the model architecture  
```

Choose where the model should be saved and how it will be named. You can also start the training from another model.
Finally, you can choose the batch size which indicates how 
```python 
models_path = './models/' # the directory in which your model weights will be saved  
model_prefix = '/yes_homography_' # the way we name the created models (we add "XXepochs"" at the end)  
epochs_already_trained = 105 # if we want to train from an already existing model, this is the way to tell at which epoch (saved every 5 epochs)
batch_size = 16 
``` 

The variables are here filled with template values, but you need to enter yours.  
Once it's done, launch this code. You'll see the losses (train and test) at each epoch.

## Dataset

The Swimm_400 dataset is available at this link : [Swimm_400](https://drive.google.com/drive/folders/1CY7x1tKAaigZabdDF99cpQVAxBDhoiq0?usp=sharing)

## Displays

2 curves appear when you train a model :
 * mAP : a metric usefull to measur the precision (#truepositives / #positives) => you want to maximize it
 * train / test loss : they need to be minimized. With few data, the test loss will eventually stop lowering => when this happens it means you're overfitting.
Basically, continue the training as long as the mAP curve grows and the test loss gets lower.
