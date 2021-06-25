# Setup

Install the requirements. If some are missing, don't hesitate to add them to the list.  
The project has originally been done in python 3.8

# training

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

## Displays

2 curves appear when you train a model :
 * mAP : a metric usefull to measur the precision (#truepositives / #positives) => you want to maximize it
 * train / test loss : they need to be minimized. With few data, the test loss will eventually stop lowering => when this happens it means you're overfitting.
Basically, continue the training as long as the mAP curve grows and the test loss gets lower.
