import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model('inceptionv3_trained_model.h5')

# Print the model summary to get an overview of the model architecture
print(model.summary())

# Get the desired layer for filter visualization
layer_index = 1  # Adjust this value based on the layer you want to visualize
layer = model.layers[layer_index]

# Retrieve the filters (weights) from the layer
filters, biases = layer.get_weights()

# Print the name and shape of the filters
print("Layer Name:", layer.name)
print("Filter Shape:", filters.shape)

# Visualize the filters
fig, axes = plt.subplots(nrows=filters.shape[3], ncols=filters.shape[2])

for i, ax in enumerate(axes.flat):
    if i < filters.shape[3]:
        # Extract the filter weights for each channel
        filter_weights = filters[:, :, :, i]
        ax.imshow(filter_weights[:, :, 0], cmap='gray')
        ax.axis('off')

plt.show()


