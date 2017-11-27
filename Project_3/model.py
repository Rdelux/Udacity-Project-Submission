import csv
import cv2
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sklearn
from sklearn.utils import shuffle

samples = []
correction = 0.2

print("Reading Data ...")
with open('./Data1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Split 20% of the samples to validation samples
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Python generator for reading image files in batches
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: 
        shuffle(samples)
        print("Reading images ...")
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                # Center Image 
                image = cv2.imread(batch_sample[0])
                angle = float(batch_sample[3])
                images.append(image)
                angles.append(angle)
                # Left Image
                image = cv2.imread(batch_sample[1])
                angle = float(batch_sample[3]) + correction
                images.append(image)
                angles.append(angle)
                # Right Image
                image = cv2.imread(batch_sample[2])
                angle = float(batch_sample[3]) - correction
                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model

row, col, ch = 160, 320, 3

# Model based on NVIDIA architecture
# Source:  https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(row,col,ch), output_shape=(row,col,ch)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

print("Training Started...")

# Mean Square Error function for regression network - different from cross-entropy fnt
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')