# main.py
from model import create_model
from tensorflow.keras.optimizers import Adam
import numpy as np

# Create model
input_shape = (your_input_shape)
num_classes = (your_num_classes)
model = create_model(input_shape, num_classes)

# Compile the model (same as during training)
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load the trained weights
model.load_weights("model_weights.h5")

# Use the model for predictions on new data
new_data = np.random.rand(10, *input_shape)
predictions = model.predict(new_data)
