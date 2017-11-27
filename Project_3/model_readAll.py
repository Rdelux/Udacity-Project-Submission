import csv
import cv2
import numpy as np
import h5py
import matplotlib.pyplot as plt

count = 0
lines = []
print("Importing Dataset ...")
with open('./Data1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        count += 1

# Read in images and steering measurement from data and append them into an array
images = []
measurements = []
correction = 0.2

print("There are {0} samples".format(count))
print("Reading images ...")
for line in lines:
    # Center image
    image = cv2.imread(line[0])
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    # Left image
    image = cv2.imread(line[1])
    images.append(image)
    measurement = float(line[3]) + correction
    measurements.append(measurement)
    # Right image
    image = cv2.imread(line[2])
    images.append(image)
    measurement = float(line[3]) - correction
    measurements.append(measurement)    
    
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model

# NVIDIA Architecture
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
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

print("Training Started ...")
# Run model - 20% of the dataset will be used for validation
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)


model.save('model.h5')