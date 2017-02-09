# Behavior Cloning using Keras

Build (and train), a convolution neural network in Keras that predicts steering angles from images to allow successful driving of a vehicle around a simulated track without leaving the road.

###Credits

- Udacity: Self-Driving Car Nano Degree
- VGG-16: Very Deep Convolutional Networks for Large-Scale Image Recognition:: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
- NVIDIA: CNN architecture - End to End Learning for Self-Driving Cars:: https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
- Keras preprocessing.image:: https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
- Vivek Yadav: An augmentation based deep neural network approach to learn human driving behavior:: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
- Matt Harvert: Training a deep learning model to steer a car in 99 lines of code:: https://hackernoon.com/training-a-deep-learning-model-to-steer-a-car-in-99-lines-of-code-ba94e0456e6a#.q19v474ta

###Setup

- System setup: https://github.com/udacity/CarND-Term1-Starter-Kit
- Sample training data: https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip
- Simulator MACOSX: https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip

###Represenative test video for track one:

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/Ra-MEWdlCWA/0.jpg)](https://www.youtube.com/watch?v=Ra-MEWdlCWA)
###Model Architecture and Training Strategy

####1. VGG-16 inspired model architecture has been used

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 29-42) 

The model includes ELU layers to introduce nonlinearity (ex. code line 29), and the data is normalized in the model using a Keras batch normalization layer (code line 27). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 48, and 51). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 55).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, left camera imagery and right camera imagery...

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to tailor VGG-16 & NVIDIA's implementation for on-road steering angle predication to better suite our use case.

My first step was to use a convolution neural network model similar to the VGG-16. I thought this model might be appropriate because its a relatively easier model architecture to understand and modify - and does a good job with image analysis.

In order to gauge how well the model was working, I added an additoinal validation set. I found that my initial models had increasing mean square error on validation data. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that my mean square error on validation error gradaully reduces. 

Interestingly, I notice that my mean square error is lower on my validation data than on my training data. Hence I am quite confident that I am avoiding overfitting. The possible reason for mean square error being lower on my validation data could be because I am not adding the augmented image data to my validation dataset.

-------------------------------------------------------------------------------------
Epoch 1/3
25600/25600 [==============================] - 272s - loss: 0.0926 - val_loss: 0.0490

Epoch 2/3
25600/25600 [==============================] - 268s - loss: 0.0382 - val_loss: 0.0359

Epoch 3/3
25600/25600 [==============================] - 269s - loss: 0.0359 - val_loss: 0.0303

-------------------------------------------------------------------------------------
The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track such as 
- near to the tree (due to the shadow), 
- near the steep right turn close to the water (due to strong correction need),
- near the open area that drives into a dirt track (due to non-clear road track)

the  to improve the driving behavior in these cases, I generated new images
- with randomly simulated shadows (helped with the tree)
- with randomly varied brightness (helped model in general)
- added left and right camera images (with +0.3 and -0.3 correction to steering angles) - (to help with steep left and right turns)

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes 

![model architecture] (https://docs.google.com/drawings/d/1glfZJVpTOzPiCq4EVRtup1TaidIDVGetRC1Q5U4adMk/pub?w=1402&h=416)

-------------------------------------------------------------------------------------
Layer (type):             Output Shape             

batchnormalization_1:      (None, 100, 100, 3)   
convolution2d_1:           (None, 98, 98, 32)                
maxpooling2d_1:            (None, 49, 49, 32)                   
convolution2d_2:           (None, 47, 47, 32)            
maxpooling2d_2:            (None, 23, 23, 32)              
convolution2d_3:           (None, 21, 21, 32)         
maxpooling2d_3:            (None, 10, 10, 32)      
convolution2d_4:           (None, 8, 8, 64)             
maxpooling2d_4:            (None, 4, 4, 64)                  
convolution2d_5:           (None, 2, 2, 128)           
maxpooling2d_5:            (None, 1, 1, 128)           
flatten_1:                 (None, 128)             
dense_1:                   (None, 1024)                 
dropout_1:                 (None, 1024)                      
dense_2:                   (None, 512)                      
dropout_2:                 (None, 512)                    
dense_3:                   (None, 1)                             

-------------------------------------------------------------------------------------


####3. Creation of the Training Set & Training Process

Since I did not have access to a joystick (required for smoother data collection) - I trained my model over Udacity's sample dataset. I generated additional datasets by applying image augmentation techniques. Here is an example image of center lane driving:

- center camera

![model architecture] (https://github.com/aksgoel/P3_Behavior_Cloning/blob/master/Steering_Images/center_2016_12_01_13_43_28_912.jpg)

To improve left and right turn coorection, I used images from left and right camera of car. Here are examples of left and right camera images:

- left camera

![model architecture] (https://github.com/aksgoel/P3_Behavior_Cloning/blob/master/Steering_Images/left_2016_12_01_13_43_28_912.jpg)

- right camera

![model architecture] (https://github.com/aksgoel/P3_Behavior_Cloning/blob/master/Steering_Images/right_2016_12_01_13_43_28_912.jpg)

- center camera in BGR

![center camera image] (https://github.com/aksgoel/P3_Behavior_Cloning/blob/master/Steering_Images/base_BGR.png)

To augment the data set with additional training data, I futher randomly applied:

- brightness shifts: (to improve model predictions in case of varying brightness leverls)

![model architecture] (https://github.com/aksgoel/P3_Behavior_Cloning/blob/master/Steering_Images/brightness_1.png)
![model architecture] (https://github.com/aksgoel/P3_Behavior_Cloning/blob/master/Steering_Images/brightness_2.png)
![model architecture] (https://github.com/aksgoel/P3_Behavior_Cloning/blob/master/Steering_Images/brightness_3.png)

- added shadows: (to improve model predictions in case of random shadows)

![model architecture] (https://github.com/aksgoel/P3_Behavior_Cloning/blob/master/Steering_Images/shadow_1.png)
![model architecture] (https://github.com/aksgoel/P3_Behavior_Cloning/blob/master/Steering_Images/shadow_2.png)
![model architecture] (https://github.com/aksgoel/P3_Behavior_Cloning/blob/master/Steering_Images/shadow_31.png)

- Resized Image:

![model architecture] (https://github.com/aksgoel/P3_Behavior_Cloning/blob/master/Steering_Images/resized.png)

- PIL format: (required to leverage keras image preprocessing tools)

![model architecture] (https://github.com/aksgoel/P3_Behavior_Cloning/blob/master/Steering_Images/PIL_format.png)

- shear shift:

![model architecture] (https://github.com/aksgoel/P3_Behavior_Cloning/blob/master/Steering_Images/shear_1.png)
![model architecture] (https://github.com/aksgoel/P3_Behavior_Cloning/blob/master/Steering_Images/sheer_2.png)
![model architecture] (https://github.com/aksgoel/P3_Behavior_Cloning/blob/master/Steering_Images/sheer_3.png)

- flipped image: image is flipped as the dataset is inbalanced, having larger left turns. 

![model architecture] (https://github.com/aksgoel/P3_Behavior_Cloning/blob/master/Steering_Images/flip.png)

With continous generation of additional training data (with image augmentation techniques), I created a large number of data points. I further applied image normalization to this dataset.

I created a validation data set without applying image augmentation techniques to it.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3. I used an adam optimizer so that manually training the learning rate wasn't necessary.

####4. Conclussion and Next Steps:

This was a very interesting project where I was able to get inspired from existing neural network models such as VGG-16, and apply a modified versions of it to a behavior cloning project using Keras. 
- The final model works fairly well on track one. As next steps I will like to enhance my applied image pre-processing techqniues and improve the model so that this works well on both Track two and on Udacity's on-road challenge 2 dataset. 
- I will also like to experiment with neural network layers such as LSTM to enhance prediction over time series. This will be especially helpful while predicting throttle values in addition to steering angle. 
