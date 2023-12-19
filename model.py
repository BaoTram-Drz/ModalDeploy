# model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

def create_model(input_shape, num_classes):
    model = Sequential()
    # Build your model architecture here
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    return model
