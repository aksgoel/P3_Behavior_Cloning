"""*****************CREDITS**********************

Udacity: Self-Driving Car Nano Degree

VGG-16: Very Deep Convolutional Networks for Large-Scale Image Recognition:: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
NVIDIA: CNN architecture - End to End Learning for Self-Driving Cars:: https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

Keras preprocessing.image:: https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
Vivek Yadav: An augmentation based deep neural network approach to learn human driving behavior:: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
Matt Harvert: Training a deep learning model to steer a car in 99 lines of code:: https://hackernoon.com/training-a-deep-learning-model-to-steer-a-car-in-99-lines-of-code-ba94e0456e6a#.q19v474ta

"""
"""*****************LIBRARIES**********************"""
import json, csv, PIL, random, cv2, numpy as np

from keras.models import load_model, Sequential, model_from_json
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Flatten, LSTM
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import img_to_array, flip_axis, random_shift, random_shear

"""*****************MODEL DEFINITION**********************"""
def model(load, shape):
    """Return a model from file or to train on."""
    if load: return load_model('model.h5')

    model = Sequential()

    model.add(BatchNormalization(input_shape= shape))

    model.add(Convolution2D(32, 3, 3, activation='elu'))
    model.add(MaxPooling2D())

    model.add(Convolution2D(32, 3, 3, activation='elu'))
    model.add(MaxPooling2D())

    model.add(Convolution2D(32, 3, 3, activation='elu'))
    model.add(MaxPooling2D())

    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(MaxPooling2D())

    model.add(Convolution2D(128, 3, 3, activation='elu'))
    model.add(MaxPooling2D())

    model.add(Flatten())


    model.add(Dense(1024, activation='elu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, activation='elu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer="adam")

    return model


"""*****************DATA GENERATION**********************"""
def get_data_X_y(data_dir, X=[], y=[]):
    """Read the log file and turn it into X/y pairs."""
    with open(data_dir + 'driving_log.csv') as fin:
        next(fin)
        log = list(csv.reader(fin))

    for row in log:
        if float(row[6]) < 20: continue  # throw away low-speed samples
        X += [row[0].strip(), row[1].strip(), row[2].strip()]       #using center, left and right images
        y += [float(row[3]), float(row[3]) + 0.3, float(row[3]) - 0.3]      #add a small angle to the left camera and subtract a small angle from the right camera

    return X, y


def process_image(path, steering_angle, augment=False, shape=(100, 100, 3)):
    """Process the image."""

    image = cv2.imread(path)       #read using opencv
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)    #convert to RGB
    image = image[45:140, 0:320]      #crop

    #augment brightness of camera images
    if augment and random.random() < 0.5:
        image = augment_brightness_camera_images(image)

    #add random shadows
    if augment and random.random() < 0.5:
        image = add_random_shadow(image)

    image = cv2.resize(image, (shape[0], shape[1]))     #resize to shape accepted by model

    #convert to numpy_array before using keras image preprocessing functions
    image = img_to_array(image)

    #Random shear, spatial and mirror image shifts using keras preprocessing functions
    if augment:
        #Random spatial vertical shifts
        image = random_shift(image, 0, 0.15, 0, 1, 2)

        #Random image flipping accross vertical axis
        if random.random() < 0.5:
            image = flip_axis(image, 1)
            steering_angle = -steering_angle

        #Random shear
        if random.random() < 0.5:
            image = random_shear(image, 0.15, 0, 1, 2, 'wrap')

    image = (image / 255. - .5).astype(np.float32)

    return image, steering_angle

def augment_brightness_camera_images(image):
    """Given an image (from Image.open), randomly augement the brightness for the image."""
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = 0.25 + np.random.uniform()
    image[:,:,2] = image[:,:,2]*random_bright
    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    return image

def add_random_shadow(image):
    """Given an image (from Image.open), randomly add a shadow to it."""
    w, h, ch = image.shape
    # Make a random box.
    x1, y1 = random.randint(0, w), random.randint(0, h)
    x2, y2 = random.randint(x1, w), random.randint(y1, h)

    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]

    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]

    shadow_mask[((X_m-x2)*(y2-y1) - (x2 - x1)*(Y_m-y1) >=0)]=1
    random_bright = 0.5

    if np.random.randint(2)==1:
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright

    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

def _generator(batch_size, X, y):
    """Generate batches of training data forever."""
    while 1:
        batch_X, batch_y = [], []
        for i in range(batch_size):
            #randomly choose sample data
            sample_index = random.randint(0, len(X) - 1)
            steering_angle = y[sample_index]

            #pre-process sample data
            image, steering_angle = process_image(X[sample_index], steering_angle, augment=True)

            #add processed sample image along with steering angle to batch
            batch_X.append(image)
            batch_y.append(steering_angle)
        yield np.array(batch_X), np.array(batch_y)


"""*****************TRAIN**********************"""

def train():
    """Load our network and our data, fit the model, save it."""
    #Load model
    net = model(load=False, shape=(100, 100, 3))

    #Load data
    X, y = get_data_X_y('./data/')

    #Print model summary
    net.summary()

    #Fit model: set number of epochs, samples per epoch
    net.fit_generator(_generator(256, X, y), samples_per_epoch=25600, nb_epoch=3)

    #Save model weights
    net.save('model.h5')

    #Save model architecture as JSON
    json_string = net.to_json()
    with open('model.json', 'w') as f:
        json.dump(json_string, f)

if __name__ == '__main__':
    train()
