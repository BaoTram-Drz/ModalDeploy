# train.py
from data import load_data
from model import create_model
from tensorflow.keras.optimizers import Adam

# Load data
file_path = "model.hdf5"
X_train, y_train, X_val, y_val = load_data(file_path)

# Create model
input_shape = X_train.shape[1:]
num_classes = len(np.unique(y_train))
model = create_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Save the model weights to an HDF5 file
model.save_weights("model_weights.h5")
