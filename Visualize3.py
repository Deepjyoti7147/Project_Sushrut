import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the saved model
model = tf.keras.models.load_model('BreastCancerDetector2.h5')

# Define the indices of the convolutional layers to visualize
conv_layer_indices = [2, 3, 5]

# Get the outputs of the specified convolutional layers
outputs = [model.layers[i].output for i in conv_layer_indices]

""" Create a new model that takes the same inputs as the original model and outputs the feature maps of the specified 
layers"""
model_short = tf.keras.models.Model(inputs=model.inputs, outputs=outputs)

# Print the summary of the new model
print(model_short.summary())

# Load and preprocess the input image
image_path = 'TestImages/btest3.jpg'
img = cv2.imread(image_path)
img = cv2.resize(img, (224, 224))
input_data = np.array([img])

# Get the feature maps of the specified layers for the input image
feature_outputs = model_short.predict(input_data)

# Set the number of columns and rows for visualization
columns = 8
rows = 8

# Visualize the feature maps for each layer
for layer_index, ftr in enumerate(feature_outputs):
    fig = plt.figure(figsize=(12, 12))
    for i in range(1, columns * rows + 1):
        fig = plt.subplot(rows, columns, i)
        fig.set_xticks([])
        fig.set_yticks([])
        plt.imshow(ftr[0, :, :, i - 1], cmap='Greys')
    plt.show()
