# **Behavioral Cloning**


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

**Click on the image to go to the youtube video**

[![IMAGE ALT TEXT](http://img.youtube.com/vi/QCjtrLyN2-w/0.jpg)](http://www.youtube.com/watch?v=QCjtrLyN2-w "Second track run outside")



[//]: # (Image References)

[histo1]: ./examples/histo1.png
[histo2]: ./examples/histo2.png
[histo3]: ./examples/histo3.png
[train1]: ./examples/train1.png
[train2]: ./examples/train2.png
[img_augmentation]: ./examples/img_augmentation.png
[sample_img]: ./examples/sample_img.png

## Rubric Points


### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* P3-Behavioral-Cloning.ipynb contains all the code to create the dataset and train the model
* drive.py for driving the car in autonomous mode
* drive_slow.py for driving the car at a slower speed in autonomous mode (specially useful for track 2)
* model.h5 containing a trained convolution neural network capable of completing both track #1 and track #2
* This writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of modification of the Nvidia architecture where I added batch normalization, dropout and Leaky relu activations (see below)

I added batch normalization layers to prevent gradient vanishing problems and try to avoid overfitting and overall have a model with a more suitable internal structure. The reason to add dropout layers was to reduce the overfitting of the training data. Finally, I chose leaky relu to add non linearities to the model in both the positive and negative activation regions of each neuron, this will also prevent 'dead neurons' and allow the system to learn better.

I acknowledge that it is a big model but I wanted to explore the possibilities that Keras offered and the different types of layers.
```
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_2 (Lambda)                (None, 160, 320, 3)   0           lambda_input_2[0][0]
____________________________________________________________________________________________________
cropping2d_2 (Cropping2D)        (None, 65, 320, 3)    0           lambda_2[0][0]
____________________________________________________________________________________________________
convolution2d_6 (Convolution2D)  (None, 31, 158, 24)   1824        cropping2d_2[0][0]
____________________________________________________________________________________________________
batchnormalization_10 (BatchNorm (None, 31, 158, 24)   96          convolution2d_6[0][0]
____________________________________________________________________________________________________
leakyrelu_6 (LeakyReLU)          (None, 31, 158, 24)   0           batchnormalization_10[0][0]
____________________________________________________________________________________________________
convolution2d_7 (Convolution2D)  (None, 14, 77, 36)    21636       leakyrelu_6[0][0]
____________________________________________________________________________________________________
batchnormalization_11 (BatchNorm (None, 14, 77, 36)    144         convolution2d_7[0][0]
____________________________________________________________________________________________________
leakyrelu_7 (LeakyReLU)          (None, 14, 77, 36)    0           batchnormalization_11[0][0]
____________________________________________________________________________________________________
convolution2d_8 (Convolution2D)  (None, 5, 37, 48)     43248       leakyrelu_7[0][0]
____________________________________________________________________________________________________
batchnormalization_12 (BatchNorm (None, 5, 37, 48)     192         convolution2d_8[0][0]
____________________________________________________________________________________________________
leakyrelu_8 (LeakyReLU)          (None, 5, 37, 48)     0           batchnormalization_12[0][0]
____________________________________________________________________________________________________
convolution2d_9 (Convolution2D)  (None, 3, 35, 64)     27712       leakyrelu_8[0][0]
____________________________________________________________________________________________________
batchnormalization_13 (BatchNorm (None, 3, 35, 64)     256         convolution2d_9[0][0]
____________________________________________________________________________________________________
leakyrelu_9 (LeakyReLU)          (None, 3, 35, 64)     0           batchnormalization_13[0][0]
____________________________________________________________________________________________________
convolution2d_10 (Convolution2D) (None, 1, 33, 64)     36928       leakyrelu_9[0][0]
____________________________________________________________________________________________________
batchnormalization_14 (BatchNorm (None, 1, 33, 64)     256         convolution2d_10[0][0]
____________________________________________________________________________________________________
leakyrelu_10 (LeakyReLU)         (None, 1, 33, 64)     0           batchnormalization_14[0][0]
____________________________________________________________________________________________________
flatten_2 (Flatten)              (None, 2112)          0           leakyrelu_10[0][0]
____________________________________________________________________________________________________
dense_6 (Dense)                  (None, 1000)          2113000     flatten_2[0][0]
____________________________________________________________________________________________________
batchnormalization_15 (BatchNorm (None, 1000)          4000        dense_6[0][0]
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 1000)          0           batchnormalization_15[0][0]
____________________________________________________________________________________________________
dense_7 (Dense)                  (None, 100)           100100      dropout_5[0][0]
____________________________________________________________________________________________________
batchnormalization_16 (BatchNorm (None, 100)           400         dense_7[0][0]
____________________________________________________________________________________________________
dropout_6 (Dropout)              (None, 100)           0           batchnormalization_16[0][0]
____________________________________________________________________________________________________
dense_8 (Dense)                  (None, 50)            5050        dropout_6[0][0]
____________________________________________________________________________________________________
batchnormalization_17 (BatchNorm (None, 50)            200         dense_8[0][0]
____________________________________________________________________________________________________
dropout_7 (Dropout)              (None, 50)            0           batchnormalization_17[0][0]
____________________________________________________________________________________________________
dense_9 (Dense)                  (None, 10)            510         dropout_7[0][0]
____________________________________________________________________________________________________
batchnormalization_18 (BatchNorm (None, 10)            40          dense_9[0][0]
____________________________________________________________________________________________________
dropout_8 (Dropout)              (None, 10)            0           batchnormalization_18[0][0]
____________________________________________________________________________________________________
dense_10 (Dense)                 (None, 1)             11          dropout_8[0][0]
====================================================================================================
Total params: 2,355,603
Trainable params: 2,352,811
Non-trainable params: 2,792
_____________________________
```

I also trained a much more simpler network (although with worse results) based on the one described in the [Comma.ai github](https://github.com/commaai/research/blob/master/train_steering_model.py)

```
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_4 (Lambda)                (None, 160, 320, 3)   0           lambda_input_4[0][0]
____________________________________________________________________________________________________
convolution2d_12 (Convolution2D) (None, 40, 80, 16)    3088        lambda_4[0][0]
____________________________________________________________________________________________________
elu_1 (ELU)                      (None, 40, 80, 16)    0           convolution2d_12[0][0]
____________________________________________________________________________________________________
convolution2d_13 (Convolution2D) (None, 20, 40, 32)    12832       elu_1[0][0]
____________________________________________________________________________________________________
elu_2 (ELU)                      (None, 20, 40, 32)    0           convolution2d_13[0][0]
____________________________________________________________________________________________________
convolution2d_14 (Convolution2D) (None, 10, 20, 64)    51264       elu_2[0][0]
____________________________________________________________________________________________________
flatten_3 (Flatten)              (None, 12800)         0           convolution2d_14[0][0]
____________________________________________________________________________________________________
dropout_9 (Dropout)              (None, 12800)         0           flatten_3[0][0]
____________________________________________________________________________________________________
elu_3 (ELU)                      (None, 12800)         0           dropout_9[0][0]
____________________________________________________________________________________________________
dense_11 (Dense)                 (None, 512)           6554112     elu_3[0][0]
____________________________________________________________________________________________________
dropout_10 (Dropout)             (None, 512)           0           dense_11[0][0]
____________________________________________________________________________________________________
elu_4 (ELU)                      (None, 512)           0           dropout_10[0][0]
____________________________________________________________________________________________________
dense_12 (Dense)                 (None, 1)             513         elu_4[0][0]
====================================================================================================
Total params: 6,621,809
Trainable params: 6,621,809
Non-trainable params: 0
```

#### 2. Attempts to reduce overfitting in the model

As mentioned in the previous section, I used dropout to reduce overfitting

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually .

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road together with data augmentation to make the model more robust

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to copy nvidia's model and play with it to understand the capabilities of the different layers that keras offers. For example, i discovered that by swapping the orignal relu activations by leaky relus, I was able to obtain better results on the track. Another key point was the addition of dropout to reduce overfitting and batch normalization for a faster training.

The first step was to collect data from the tracks. I collected multiple runs on track #1, including recoveries, clockwise and counterclockwise driving. This was enough to complete track #1 with my code but it was not enough to complete track #2. So, in order to finish both tracks, I improved the model and drove 1 lap on track #2 to collect some data and teach the models the key features of this new environment.

One of the key challenges that I set myself was to use a uniform dataset.
The process to do so is as follows:
* Load center, right and left images to my dataset
* Calculate the corresponding angles for the left and right images (I used +- 0.25 steering units for this)
* Flip the center images and calculated the 'inverse angle'
* Generate a copy of each image that is rotated and has different brightness

After this step I get a histogram similar to this one:

![histo1]

I decided that I would have a uniform data distribution with a sample number (per 0.05 steering unit bin) equal to the average number of samples in the original distribution. The first step towards this is to randomly drop images that are in the most populated bins(cell #7). This process is done using the `calc_bin_drop_percentage`, and `drop_imgs` functions:
* `calc_bin_drop_percentage`: calculates how many images do we need to drop on each bin to reduce their number to the average count number for the dataset
* `drop_imgs`: calculates which images need to be dropped to achieve the desired distribution



![histo2]

Now I need to raise the number of samples in the underpopulated bins. To achieve this, I use the `calc_imgs_to_gen`, `gen_extra_angles`, `find_appropiated_new_angle`:
* `calc_imgs_to_gen`: calculates how many images do we need to generate in order to fill every bin up to the average count number for the whole dataset
* `gen_extra_angles`: translate imgs and gen new angles (uniform - ish)
* `find_appropiated_new_angle`: Each bin needs to get to the total average in order to get a uniform distribution generates an angle that belongs to a bin that has not reached the target number yet gets an angle and tries to find a new angle (and image) to generate that is inside a certain range i.e. you dont want to generate a -1 angle from a +1 angle image (too much translation).
This is by no means an optimal implementation but it kindof works ;)

![histo3]
Green = dropped distribution
Red = New samples generated
Blue = Resultant 'uniform distribution'

In order to calculate properly the drop probabilities (`calc_bin_drop_percentage`) and number of images to generate (`calc_imgs_to_gen`), I perform an initial data exploration using just the angle data. This is done to prevent loading all the images in memory and allows us to achieve the results that we want even when working in individual batches in the `generator`

It is important to remark that this is not the ideal way of doing this, this is just how things turned out after many iterations. Ideally, if I had to do this again, I would first process all the data in a similar way and then store it in a pickle file, that would significantly reduce the training time and the complexity of the generator

After all this processing we get a dataset that is approximately twice the size than the original one.

Here you can see an example of the result of the data augmentation:

![img_augmentation]

Then, as mentioned before, I use a `generator` to pass all the data to the model and avoid loading all the images into memory at once.

Then, my model is trained:

![train1]

and the Comma.ai model too:

![train2]

As we can see, the models could perform very well with just a few epochs.

The final step was to run the simulator to see how well the car was driving around the tracks. Both models are capable of finishing track #1 but only my model is capable of finishing track #2 (see videos in the repo or [youtube](http://www.youtube.com/watch?v=QCjtrLyN2-w).

**Click on the images to go to the youtube videos:**
  
Second track from the outside
[![IMAGE ALT TEXT](http://img.youtube.com/vi/QCjtrLyN2-w/0.jpg)](http://www.youtube.com/watch?v=QCjtrLyN2-w "Second track run outside")


Second track from the car front camera (low res)
[![IMAGE ALT TEXT](http://img.youtube.com/vi/3VOWl3mOm6g/0.jpg)](http://www.youtube.com/watch?v=3VOWl3mOm6g "Second track run outside")

First track from the car front camera (low res)
[![IMAGE ALT TEXT](http://img.youtube.com/vi/KFUdqSXg424/0.jpg)](http://www.youtube.com/watch?v=KFUdqSXg424 "Second track run outside")
