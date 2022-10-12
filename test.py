import os
import sys
import random
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 50

from keras.models import load_model
model = load_model('best_model.hdf5')


def predictImage(filename):
    img = cv2.imread(filename)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    plt.imshow(img)

    Y = np.array(img)
    X = np.expand_dims(Y, axis=0)
    val = model.predict(X)
    print(val)
    if val < 50:
        return  plt.xlabel("Car Severly Damaged", fontsize=30)
    elif val >= 50:
        return plt.xlabel("Car has Moderate or No damage", fontsize=30)


fn="data/data1a/validation/00-damage/0010.JPEG"

predictImage(fn)