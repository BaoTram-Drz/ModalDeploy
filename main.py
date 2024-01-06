import h5py
import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing import image
from PIL import Image
from tensorflow.keras.models import model_from_json

model_path = 'model.hdf5'  
# Load the Keras model from the HDF5 file
def load_keras_model(file_path):
    try:
        with h5py.File(file_path, 'r') as file:
            model_json = file.attrs['model_config']
            model = model_from_json(model_json)
            model.load_weights(file_path)
        return model
    except Exception as e:
        print(f"Error loading the model: {str(e)}")
        return None

# Hàm dự đoán hình ảnh
def predict_image(model, img_array):
    # Chuẩn hóa hình ảnh
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    # Dự đoán
    predictions = model.predict(img_array)
    return predictions

# Giao diện Streamlit
def main():
    st.title("Ứng dụng: Dự đoán bệnh của lúa nước!")
    # Tải mô hình
    loaded_model = load_keras_model(model_path)

    if loaded_model is None:
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
            predictions = predict_image(loaded_model, img_array)
            st.write("Kết quả dự đoán:", predictions)
            st.write("0: Bệnh vàng lá chín sớm")
            st.write("1: Bệnh cháy bìa lá")
            st.write("2: Bệnh lem lép hạt")
            st.write("3: Bệnh đạo ôn")
            st.write("4: Bệnh đốm nâu")
            st.write("5: Bệnh đạo ôn cổ bông")
            st.write("6: Bệnh bạc lá")
            st.write("7: Bọ xít")
            st.write("8: Bình thường")
            st.write("9: Bệnh tungro/ vàng lùn lùn xoắn lá")

        
        
if __name__ == "__main__":
    main()