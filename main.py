
import streamlit as st
import h5py
from tensorflow.keras.models import load_model

# Đường dẫn tới tệp model.hdf5
model_path = "model.hdf5"

# Hàm để mở tệp và trả về đối tượng model
def load_model():
    model = load_model(model_path)  # Sử dụng hàm load_model của Keras để tải mô hình
    return model

# Load model
model = load_model()

# Giao diện người dùng
st.title("Ứng dụng Streamlit với model.hdf5")
# TODO: Thêm các thành phần giao diện khác và sử dụng model
