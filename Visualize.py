import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as K
# Load the saved model

model = tf.keras.models.load_model('BreastCancerDetector2.h5')


print(model.summary())
layers = model.layers
filters, biases = model.layers[1].get_weights()
print(layers[1].name,filters.shape)


fig1 = plt.figure(figsize=(8,12))
columns = 8
rows = 8
n_filters = columns * rows
for i in range(1,n_filters + 1):
    f = filters[:, :, :, i-1]
    fig1 = plt.subplot(rows, columns,i)
    fig1.set_xticks([])
    fig1.set_yticks([])
    plt.imshow(f[:, :,0], cmap='gray')
plt.show()


conv_layer_index = [2, 3, 5, 6, 8, 9]
outputs = [model.layers[i].output for i in conv_layer_index]
model_short = Model(inputs=model.inputs, outputs=outputs)
print(model_short. summary())

# Taking Image and preprocessing it

image_path = 'TestImages/caratest2.jpeg'
img = cv2.imread(image_path)
img = cv2.resize(img, (224, 224))
input_data = np.array([img])

feature_output = model_short.predict(input_data)

columns = 8
rows = 8

for ftr in feature_output:
    fig = plt.figure(figsize=(12,12))
    for i in range(1,columns*rows + 1):
        fig = plt.subplot(rows,columns,i)
        fig.set_xticks([])
        fig.set_yticks([])
        plt.imshow(ftr[0, :, :, i-1], cmap='Blues')
    plt.show()