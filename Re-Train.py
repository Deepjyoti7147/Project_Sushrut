import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

# Load and preprocess the images
X = []
y = []

train_dir = 'ultrasound breast classification/train'
val_dir = 'ultrasound breast classification/val'

for folder in tqdm(os.listdir(train_dir)):
    if folder == 'benign':
        label = 0  # benign class label
    else:
        label = 1  # malignant class label

    folder_path = os.path.join(train_dir, folder)
    for file in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, file))
        img = cv2.resize(img, (224, 224))
        X.append(img)
        y.append(label)

for folder in tqdm(os.listdir(val_dir)):
    if folder == 'benign':
        label = 0  # benign class label
    else:
        label = 1  # malignant class label

    folder_path = os.path.join(val_dir, folder)
    for file in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, file))
        img = cv2.resize(img, (224, 224))
        X.append(img)
        y.append(label)

# Convert the data to NumPy arrays
X = np.array(X)
y = np.array(y)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.utils import shuffle

# Shuffle the training data
X_train, y_train = shuffle(X_train, y_train, random_state=42)

# Convert the labels to categorical format
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Load the necessary libraries
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create the InceptionV3 model
num_classes = 2  # Number of output classes
base_model = InceptionV3(weights=None, include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define batch size and number of epochs
batch_size = 16
epochs = 10

# Create image data generators
train_datagen = ImageDataGenerator(...)
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)

valid_datagen = ImageDataGenerator(...)
valid_generator = valid_datagen.flow(X_test, y_test, batch_size=batch_size)

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // batch_size,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=len(X_test) // batch_size
)

# Save the trained model
model.save('inceptionv3_trained_model.h5')
