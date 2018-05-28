#!/usr/bin/python3
'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.applications import vgg16

# dimensions of our images.
img_width, img_height = 224, 224

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'image/image_train'
validation_data_dir = 'image/image_cv'
nb_train_ng = 170
nb_train_ok = 182
nb_validation_ng = 42
nb_validation_ok = 46
nb_train_samples = nb_train_ng + nb_train_ok
nb_validation_samples = nb_validation_ng + nb_validation_ok
epochs = 50
batch_size = 4

def save_bottleneck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    base_model = vgg16.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = base_model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = base_model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)

def train_top_model():
    train_data = np.load('bottleneck_features_train.npy')
    train_labels = np.array(
        [0] * nb_train_ng + [1] * nb_train_ok)

    validation_data = np.load('bottleneck_features_validation.npy')
    validation_labels = np.array(
        [0] * nb_validation_ng + [1] * nb_validation_ok)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=train_data.shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    top_model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    top_model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    top_model.save_weights(top_model_weights_path)

def realtimeCheck():
    import cv2
    
    base_model = vgg16.VGG16(include_top=False, weights='imagenet')

    top_model = Sequential()
    top_model.add(Flatten(input_shape=(7,7,512)))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    top_model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])
    top_model.load_weights(top_model_weights_path)
    
    datagen = ImageDataGenerator(rescale=1. / 255)

    INPUT_TITLE = 'movie06'
    INPUT_MOVIE = 'movie/' + INPUT_TITLE + '.mp4'
    OUTPUT_TITLE = INPUT_TITLE
    OUTPUT_SIZE = (224, 224)
    INTERVAL = 30  # in frame (fps = 30)

    cap = cv2.VideoCapture(INPUT_MOVIE)
    rep, frame = cap.read()

    INPUT_WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    INPUT_HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    print('input size = ', INPUT_WIDTH, INPUT_HEIGHT)
    
    i = 0
    while rep is True:
        if i % INTERVAL == 0:
            OUTPUT_NUM = int(i/INTERVAL)
            OUTPUT_FILE = 'result/' + OUTPUT_TITLE + \
                '_' + '{:0>4}'.format(OUTPUT_NUM) + '.bmp'
            frame_trimed = frame[290:1010, :]
            frame_resized = cv2.resize(frame_trimed, OUTPUT_SIZE)
            input_data = np.asarray([frame_resized, ])/255

            base_prediction = base_model.predict(input_data)
            top_prediction = top_model.predict(base_prediction)
            
            if top_prediction < 0.5:
                print('cut!')
                cv2.putText(frame_resized, 'NG!', (0, 10),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1,
                cv2.LINE_AA)
            else: 
                print('ok!')
                cv2.putText(frame_resized, 'OK', (0, 10),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1,
                cv2.LINE_AA)

            #cv2.imwrite(OUTPUT_FILE, frame_resized)
            cv2.imshow("output", frame_resized)
            cv2.waitKey(1)
        rep, frame = cap.read()
        i += 1
    
    cv2.destroyAllWindows()
    cap.release()
    
