# Convolutional Neural Network

# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
tf.__version__

# Part 1 - Data Preprocessing

# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,   #feature scaling of all the pixel values
                                   shear_range = 0.2,   #transformations
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),   #final image size when they will be fed into the convolutional NN
                                                 batch_size = 32,   # #of images in a batch size
                                                 class_mode = 'binary')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)   #only feature scaling no transformations
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()   #initializing our cnn as a layers of sequence

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))   #filters are the number of kernels, the size of one kernel is 3by3
#activation function is required till the last layer, input shape to match the target size and 3 for RGB images and 1 for BW ones

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))   

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))   #imput shape only added at the starting layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))   # #of neurons are 128

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))   #sigmoid for binary and softmax for multiclass classification

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])   #adam is the SGD

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)   #training and testing done in one step. 

# Part 4 - Making a single prediction

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))   #path of the test image to be predicted
test_image = image.img_to_array(test_image)   #converting PIL to array coz predict method expects 2D array
test_image = np.expand_dims(test_image, axis = 0)   #the images were converted into bacthes hence an additional dimension has to be added to the array to match the dimension size.
#axis=0 to specify that the dimension is the same as the first dimension
result = cnn.predict(test_image)
training_set.class_indices   #this attribute will tell which class 0 and 1 stand for
if result[0][0] == 1:   #first 0 is for the batch number and the second 0 is for the only first element i.e. our single image prediction
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)

"""
image augmentation is applying transformations on the training sets of the CNN to reduce overfitting
We augment the variations in the images

Go to keras api and to data preprocessing for reference
Try different transformations

The feature scaling is done on the test set but not transformations to avoid data leakage from the test set

As long as the output layer isn't reached it is good to add a rectifier function





"""