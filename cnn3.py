# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:35:35 2020

@author: ali hussain
"""

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import seaborn as sns

np.random.seed(2)

from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix
#import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


sns.set(style='white', context='notebook', palette='deep')


# o/p    Using TensorFlow backend
"""
2. Data preparation
2.1 Load data
# Load the data
"""

train = pd.read_csv("D:\mystuff\input\samp_train.csv")

test = pd.read_csv("D:\mystuff\input\samp_test.csv")

Y_train = train["label"]


# Drop 'label' column

X_train = train.drop(labels = ["label"],axis = 1) 

# free some space
del train 

g = sns.countplot(Y_train)

Y_train.value_counts()
"""
1    4684
7    4401
3    4351
9    4188
2    4177
6    4137
0    4132
4    4072
8    4063
5    3795
Name: label, dtype: int64

2.2 Check for null and missing values
"""
# Check the data

X_train.isnull().any().describe()

"""
count       784
unique        1
top       False
freq        784
dtype: object
"""

test.isnull().any().describe()

"""
count       784
unique        1
top       False
freq        784
dtype: object
2.3 Normalization
"""
# Normalize the data

X_train = X_train / 255.0

test = test / 255.0

"""
2.3 Reshape
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
"""

X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)

"""
2.5 Label encoding
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
"""

Y_train = to_categorical(Y_train, num_classes = 10)

#2.6 Split training and valdiation set
#Split data allowing 10% for valiation, 10% for test and 80% for training

# Split the train and the validation set for the fitting

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)

#3. CNN
#3.1 Define the model
# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

"""
WARNING:tensorflow:From /opt/conda/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
WARNING:tensorflow:From /opt/conda/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:3445: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
Instructions for updating:
Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
3.2 Set the optimizer and annealer

"""

# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
epochs = 21
batch_size = 86

#3.3 Data augmentation
# With data augmentation to prevent overfitting (accuracy 0.99286)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)

"""
Data augmentation techniques:

Randomly rotate some training images by 10 degrees
Randomly Zoom by 10% some training images
Randomly shift images horizontally by 10% of the width
Randomly shift images vertically by 10% of the height
"""
# Fit the model

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])

"""
WARNING:tensorflow:From /opt/conda/lib/python3.6/site-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
Epoch 1/21
 - 165s - loss: 0.4429 - acc: 0.8580 - val_loss: 0.0673 - val_acc: 0.9778
Epoch 2/21
 - 164s - loss: 0.1391 - acc: 0.9579 - val_loss: 0.0572 - val_acc: 0.9833
Epoch 3/21
 - 164s - loss: 0.1029 - acc: 0.9694 - val_loss: 0.0429 - val_acc: 0.9854
Epoch 4/21
 - 164s - loss: 0.0864 - acc: 0.9742 - val_loss: 0.0497 - val_acc: 0.9878
Epoch 5/21
 - 163s - loss: 0.0796 - acc: 0.9784 - val_loss: 0.0498 - val_acc: 0.9865
Epoch 6/21
 - 163s - loss: 0.0739 - acc: 0.9781 - val_loss: 0.0428 - val_acc: 0.9894
Epoch 7/21
 - 163s - loss: 0.0693 - acc: 0.9799 - val_loss: 0.0519 - val_acc: 0.9860
Epoch 8/21
 - 164s - loss: 0.0652 - acc: 0.9811 - val_loss: 0.0378 - val_acc: 0.9897
Epoch 9/21
 - 164s - loss: 0.0659 - acc: 0.9815 - val_loss: 0.0375 - val_acc: 0.9905
Epoch 10/21
 - 164s - loss: 0.0640 - acc: 0.9817 - val_loss: 0.0415 - val_acc: 0.9892
Epoch 11/21
 - 164s - loss: 0.0663 - acc: 0.9818 - val_loss: 0.0393 - val_acc: 0.9921
Epoch 12/21
 - 164s - loss: 0.0655 - acc: 0.9817 - val_loss: 0.0472 - val_acc: 0.9905
Epoch 13/21
 - 164s - loss: 0.0660 - acc: 0.9827 - val_loss: 0.0526 - val_acc: 0.9886
Epoch 14/21
 - 164s - loss: 0.0669 - acc: 0.9819 - val_loss: 0.0499 - val_acc: 0.9902

Epoch 00014: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
Epoch 15/21
 - 164s - loss: 0.0472 - acc: 0.9861 - val_loss: 0.0421 - val_acc: 0.9918
Epoch 16/21
 - 165s - loss: 0.0487 - acc: 0.9862 - val_loss: 0.0408 - val_acc: 0.9907
Epoch 17/21
 - 165s - loss: 0.0494 - acc: 0.9863 - val_loss: 0.0388 - val_acc: 0.9910

Epoch 00017: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
Epoch 18/21
 - 164s - loss: 0.0407 - acc: 0.9888 - val_loss: 0.0297 - val_acc: 0.9926
Epoch 19/21
 - 164s - loss: 0.0415 - acc: 0.9888 - val_loss: 0.0312 - val_acc: 0.9918
Epoch 20/21
 - 164s - loss: 0.0383 - acc: 0.9892 - val_loss: 0.0337 - val_acc: 0.9926
Epoch 21/21
 - 165s - loss: 0.0377 - acc: 0.9896 - val_loss: 0.0336 - val_acc: 0.9918

Epoch 00021: ReduceLROnPlateau reducing learning rate to 0.0001250000059371814.
4. Evaluate the model
4.1 Training and validation curves

# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
"""
# Evaluation
final_loss, final_acc = model.evaluate(X_test, Y_test)
print(final_loss, final_acc)

"""
4200/4200 [==============================] - 6s 2ms/step
0.025437863260102903 0.9935714285714285

"""
















