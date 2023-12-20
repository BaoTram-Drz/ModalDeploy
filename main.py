# main.py
import subprocess
import os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from model import create_model

import streamlit as st
import numpy as np
from tensorflow.keras.optimizers import Adam
 
input_shape = (64, 64, 3)
num_classes = 16
model = create_model(input_shape, num_classes)
 
for layer in model.layers:
    print(layer.name)
 
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
 
script_directory = os.path.dirname(os.path.abspath(__file__))
 
weights_file_path = os.path.join(script_directory, "model.hdf5")  # Cập nhật tên file
try:
    model.load_weights(weights_file_path, by_name=True)
except ValueError as e:
    st.error(f"Lỗi: {str(e)}")
except Exception as e:
    st.error(f"Lỗi: Đã xảy ra một lỗi không mong muốn. {e}") 
new_data = np.random.rand(10, *input_shape)
predictions = model.predict(new_data)
