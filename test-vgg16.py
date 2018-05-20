#!/sur/bin/python3
# https://www.learnopencv.com/keras-tutorial-using-pre-trained-imagenet-models/
import numpy as np
import keras
from keras.applications import vgg16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
# import matplotlib.pyplot as plt

model = vgg16.VGG16(weights='imagenet')

filename = 'image/ok0041.bmp'
original = load_img(filename, target_size=(224, 224))
print('PIL image size', original.size)

numpy_image = img_to_array(original)
print('numpy array size', numpy_image.shape)

image_batch = np.expand_dims(numpy_image, axis=0)
print('image batch size', image_batch.shape)

processed_image = vgg16.preprocess_input(image_batch.copy())
predictions = model.predict(processed_image)

label = decode_predictions(predictions)
print(label)
