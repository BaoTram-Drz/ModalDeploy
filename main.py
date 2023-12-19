# main.py
# from model import create_model
# from tensorflow.keras.optimizers import Adam
# import numpy as np

import subprocess

# Install TensorFlow
subprocess.call(['pip', 'install', 'tensorflow==2.7.0'])

# Now import the required modules
from model import create_model
import streamlit as st
import numpy as np
from tensorflow.keras.optimizers import Adam

# Create model
input_shape = (64, 64, 3)
num_classes = 10
model = create_model(input_shape, num_classes)

# Compile the model (same as during training)
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load the trained weights
model.load_weights("model_weights.h5")

# Use the model for predictions on new data
new_data = np.random.rand(10, *input_shape)
predictions = model.predict(new_data)
