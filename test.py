import tensorflow as tf
import cv2
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
# Load the saved model
model = tf.keras.models.load_model('BreastCancerDetector2.h5')

# Load and preprocess the input image
image_path = 'TestImages/caratest2.jpeg'
img = cv2.imread(image_path)
img = cv2.resize(img, (224, 224))
input_data = np.array([img])

# Make predictions
predictions = model.predict(input_data)

# Process the predictions
# ...
class_labels = ['Benign', 'Malignant']  # Define class labels

# Find the index of the maximum probability
max_prob_index = np.argmax(predictions)

# Retrieve the corresponding class label
predicted_class = class_labels[max_prob_index]

# Print the predicted class
print("The model preidicts:" , predicted_class)
# Print the predictions
print(predictions)
