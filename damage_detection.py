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

DATA_DIR = 'data/data1a'
train_dir = os.path.join(DATA_DIR, 'training/')
test_dir = os.path.join(DATA_DIR, 'validation/')

def seed_it_all(seed=7):
    tf.random.set_seed(seed)
    np.random.seed(seed)

seed_it_all()

print("seed")

train_datagen = ImageDataGenerator(rescale=1 / 255,
                                   rotation_range=40,
                                   width_shift_range=0.4,
                                   height_shift_range=0.4,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1 / 255)

train_dataset = train_datagen.flow_from_directory(train_dir,
                                                  target_size=(IMG_SIZE, IMG_SIZE),
                                                  batch_size=BATCH_SIZE,
                                                  class_mode='binary')

test_dataset = test_datagen.flow_from_directory(test_dir,
                                                target_size=(IMG_SIZE, IMG_SIZE),
                                                batch_size=BATCH_SIZE,
                                                class_mode='binary')

test_dataset.class_indices


def block(x, filters, kernel_size, repetitions, pool_size=2, strides=2):
    for i in range(repetitions):
        x = tf.keras.layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size, strides)(x)
    return x


def get_model():
    image_inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = block(image_inputs, 8, 3, 2)
    x = block(x, 16, 3, 2)
    x = block(x, 32, 3, 2)
    x = block(x, 64, 3, 2)
    x = block(x, 128, 3, 2)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    output = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=[image_inputs], outputs=[output])
    return model


model = get_model()
print("get_model")
print("block")

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, min_delta=0.001)
lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', verbose=1, patience=2, factor=0.2, min_lr=0.0001)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.hdf5', monitor='val_accuracy', verbose=1,save_best_only=True,mode='max')

callbacks = [early_stopping, lr_reduction, model_checkpoint]


history = model.fit(train_dataset,
                    validation_data=test_dataset,
                    epochs=EPOCHS,
                    callbacks=callbacks,
                    batch_size=BATCH_SIZE)


def predictImage(filename):
    img = cv2.imread(filename)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    plt.imshow(img)

    Y = np.array(img)
    X = np.expand_dims(Y, axis=0)
    val = model.predict(X)
    print(val)
    if val < 50:
        plt.xlabel("Car Damaged", fontsize=30)
    elif val >= 50:
        plt.xlabel("Car Not Damaged", fontsize=30)


predictImage("data/data1a/validation/00-damage/0006.JPEG")