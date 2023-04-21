import os.path
import h5py
import numpy as np
import tensorflow as tf

tf.config.list_physical_devices('GPU')

data_path = "dataset/gray/"

# 加载HDF5格式的数据集
with h5py.File(data_path+'face_ckp.hdf5', 'r') as f:
    face_train_data = f['face_train_data'][:]
    face_train_labels = f['face_train_labels'][:]
    face_val_data = f['face_val_data'][:]
    face_val_labels = f['face_val_labels'][:]
    face_test_data = f['face_test_data'][:]
    face_test_labels = f['face_test_labels'][:]

# 将数据集调整为适合CNN-LSTM模型的形状
face_train_data = np.reshape(face_train_data, (face_train_data.shape[0], 1, face_train_data.shape[1], face_train_data.shape[2], 1))
face_val_data = np.reshape(face_val_data, (face_val_data.shape[0], 1, face_val_data.shape[1], face_val_data.shape[2], 1))
face_test_data = np.reshape(face_test_data, (face_test_data.shape[0], 1, face_test_data.shape[1], face_test_data.shape[2], 1))

print(face_train_data.shape)
print(face_val_data.shape)
print(face_test_data.shape)

# 构建CNN-LSTM模型
model = tf.keras.Sequential([
    tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                                    input_shape=(1, 192, 192, 1)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu')),
    tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, (3, 3), activation='relu')),
    tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
    tf.keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=False),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7, activation='softmax')
])

model.summary()

# 编译模型
model.compile(loss="categorical_crossentropy", optimizer="RMSprop", metrics=['accuracy'])

# TensorBoard回调函数
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="log/CL", histogram_freq=1)

# 定义EarlyStopping回调函数
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, verbose=1)

# 训练模型
history = model.fit(face_train_data, face_train_labels, batch_size=32, epochs=100, verbose=1,
                    validation_data=(face_val_data, face_val_labels), callbacks=[tensorboard_callback, early_stop])

# 评估模型
score = model.evaluate(face_test_data, face_test_labels)
print("测试准确率：", score[1])

# 保存模型
if not os.path.exists("model"):
    os.makedirs("model")
model.save("model/CNN-LSTM_1.h5")
