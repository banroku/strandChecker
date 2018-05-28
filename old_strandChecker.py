#!/usr/bin/python3
import numpy as np
import keras
from keras.applications import vgg16
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Dropout, Flatten, Dense

base_model = vgg16.VGG16(weights='imagenet', include_top=False,
                  input_shape=(224, 224, 3))

batch_size = 16

train_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.2
        height_shift_range=0.2
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=30,
        horizontal_flip=True,
        fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'image/image_train',
        target_size=(224, 224),
        batch_size=4,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'image/image_cv',
        target_size=(224, 224),
        batch_size=4,
        class_mode='binary')

top_model = keras.Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=1,
        validation_data=validation_generator,
        validation_steps=100)
