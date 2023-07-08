import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
# import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import os
import sys
import py2exe
from distutils.core import setup
import matplotlib
matplotlib.use('TkAgg')
root = tk.Tk()
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Get the base path of the executable
base_path = getattr(sys, '_MEIPASS', os.getcwd())

# Construct the absolute path to the model file
model_path = os.path.join(base_path, 'inceptionv3_trained_model.h5')

# Load the saved model
# model = tf.keras.models.load_model()
model = load_model(model_path)
class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classification")

        # Create a label for displaying the loaded image
        self.image_label = tk.Label(root)
        self.image_label.pack()

        # Create a button to load an image
        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack()

        # Create a button for analysis
        self.analysis_button = tk.Button(root, text="Analysis", command=self.analyze_image)
        self.analysis_button.pack()

        # Create a label for displaying the prediction result
        self.result_label = tk.Label(root, text="")
        self.result_label.pack()

    def load_image(self):
        # Prompt user to select an image file
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])

        # Load and display the image
        if file_path:
            image = Image.open(file_path)
            image = image.resize((300, 300))  # Resize the image for display
            self.display_image(image)
            self.image_path = file_path

    def display_image(self, image):
        # Convert the Image object to Tkinter PhotoImage and display it on the label
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def analyze_image(self):
        if hasattr(self, 'image_path'):
            # Load and preprocess the input image
            img = cv2.imread(self.image_path)
            img = cv2.resize(img, (224, 224))
            input_data = np.array([img])

            # Make predictions
            predictions = model.predict(input_data)

            # Process the predictions
            class_labels = ['Benign', 'Malignant']  # Define class labels

            # Find the index of the maximum probability
            max_prob_index = np.argmax(predictions)

            # Retrieve the corresponding class label and probability
            predicted_class = class_labels[max_prob_index]
            probability = predictions[0][max_prob_index]

            # Display the prediction result
            self.result_label.config(text=f"Predicted Class: {predicted_class}\nProbability: {probability:.4f}")
        else:
            self.result_label.config(text="No image loaded")


gui = GUI(root)
root.mainloop()
