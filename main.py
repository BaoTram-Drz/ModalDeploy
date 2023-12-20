# main.py
# from model import create_model
# from tensorflow.keras.optimizers import Adam
# import numpy as np

import subprocess
import os


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
script_directory = os.path.dirname(os.path.abspath(__file__))

# Chỉ định đường dẫn đến file trọng số của mô hình so với thư mục của script
weights_file_path = os.path.join(script_directory, "model.hdf5")  # Cập nhật tên file

# Nạp trọng số đã được huấn luyện
model.load_weights(weights_file_path)

# Use the model for predictions on new data
new_data = np.random.rand(10, *input_shape)
predictions = model.predict(new_data)
