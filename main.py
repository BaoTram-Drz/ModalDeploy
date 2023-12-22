import streamlit as st
import h5py
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Tải mô hình từ file HDF5
model_path = 'model.hdf5'
# model = load_model(model_path)

# Hàm dự đoán hình ảnh
def predict_image(img_array):
    # Chuẩn hóa hình ảnh
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Dự đoán
    predictions = model.predict(img_array)

    return predictions

# Giao diện Streamlit
def main():
    st.title("Ứng dụng Dự đoán bệnh của lúa nước")

    # Tải lên hình ảnh từ máy tính
    uploaded_file = st.file_uploader("Tải lên hình ảnh", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Đọc hình ảnh từ file
        img = image.load_img(uploaded_file, target_size=(64, 64))
        img_array = image.img_to_array(img)

        # Hiển thị hình ảnh
        st.image(img, caption='Hình ảnh tải lên', use_column_width=True)

        # Dự đoán
        if st.button("Dự đoán"):
            predictions = predict_image(img_array)
            st.write("Kết quả dự đoán:", predictions)

if __name__ == "__main__":
    main()
