import h5py
import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing import image
from PIL import Image
from tensorflow import keras

# Tải mô hình từ tệp HDF5
def load_model_from_hdf5(file_path):
    model = None
    with h5py.File(file_path, "r") as file:
        model = file.get("default")  # Thay "model_name" bằng tên thực sự của dataset
        model = model[()] if model is not None else None
    return model

# Hàm dự đoán hình ảnh
def predict_image(model, img_array):
    # Chuẩn hóa hình ảnh
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    # Dự đoán
    predictions = model.dir(img_array)
    return predictions

# Giao diện Streamlit
def main():
    st.title("Ứng dụng Dự đoán bệnh của lúa nước")
    # Tải mô hình
    model = load_model_from_hdf5('model.hdf5')  # Thay 'model.hdf5' bằng đường dẫn thực tế đến tệp HDF5 của bạn

    if model is None:
        st.write("Không thể tải mô hình.")
        return

    # Tải lên hình ảnh từ máy tính
    uploaded_file = st.file_uploader("Tải lên hình ảnh", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Đọc hình ảnh từ tệp
        img = Image.open(uploaded_file)
        img = img.resize((256, 256))  # Resize the image to match the target size
        img_array = image.img_to_array(img)

        # Hiển thị hình ảnh
        st.image(img, caption='Hình ảnh tải lên', use_column_width=True)

        # Dự đoán
        if st.button("Dự đoán"):
            predictions = predict_image(model, img_array)
            st.write("Kết quả dự đoán:", predictions)

if __name__ == "__main__":
    main()