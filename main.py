import streamlit as st
import tensorflow as tf

def load_model():
    # Load the model from the .hdf5 file
    model = tf.keras.models.load_model('model.hdf5')
    return model

def main():
    # Load the model
    model = load_model()

    # Create a file uploader
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg'])

    if uploaded_file is not None:
        # Preprocess the image
        image = preprocess_image(uploaded_file)

        # Make predictions
        predictions = model.predict(image)

        # Display the predictions
        st.write(predictions)

if __name__ == '__main__':
    main()
