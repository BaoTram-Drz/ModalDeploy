import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import streamlit as st

# Đường dẫn tới tệp "model.hdf5"
model_path = 'model.hdf5'

# Thiết lập các thông số huấn luyện
input_shape = (64, 64, 3)
num_classes = 2
batch_size = 32
epochs = 20

# Tiền xử lý dữ liệu huấn luyện và xác thực
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
train_generator = train_datagen.flow_from_directory('training_data_directory',
                                                    target_size=input_shape[:2],
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

# Tạo mô hình và tải trọng số từ tệp "model.hdf5"
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# Thêm các lớp khác của mô hình của bạn ở đây
# ...

# Tải trọng số từ tệp "model.hdf5"
model.load_weights(model_path)

# Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(train_generator,
          steps_per_epoch=train_generator.samples // batch_size,
          epochs=epochs)

# Lưu trữ trọng số huấn luyện thành tệp "model.hdf5"
model.save_weights(model_path)

# Hiển thị thông báo hoàn thành
st.write("Huấn luyện mô hình đã hoàn thành!")
