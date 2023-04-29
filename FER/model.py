import os.path
import h5py
import numpy as np
import tensorflow as tf

data_path = "dataset/gray/"

# 加载HDF5格式的数据集
with h5py.File(data_path+'face_ckp.hdf5', 'r') as f:
    face_train_data = f['face_train_data'][:]
    face_train_labels = f['face_train_labels'][:]
    face_val_data = f['face_val_data'][:]
    face_val_labels = f['face_val_labels'][:]
    face_test_data = f['face_test_data'][:]
    face_test_labels = f['face_test_labels'][:]
with h5py.File(data_path+'eye_ckp.hdf5', 'r') as f:
    eye_train_data = f['eye_train_data'][:]
    eye_train_labels = f['eye_train_labels'][:]
    eye_val_data = f['eye_val_data'][:]
    eye_val_labels = f['eye_val_labels'][:]
    eye_test_data = f['eye_test_data'][:]
    eye_test_labels = f['eye_test_labels'][:]
with h5py.File(data_path+'mouth_ckp.hdf5', 'r') as f:
    mouth_train_data = f['mouth_train_data'][:]
    mouth_train_labels = f['mouth_train_labels'][:]
    mouth_val_data = f['mouth_val_data'][:]
    mouth_val_labels = f['mouth_val_labels'][:]
    mouth_test_data = f['mouth_test_data'][:]
    mouth_test_labels = f['mouth_test_labels'][:]
# 将数据集调整为适合CNN-LSTM模型的形状
face_train_data = np.reshape(face_train_data, (face_train_data.shape[0], face_train_data.shape[1], face_train_data.shape[2], 1))
face_val_data = np.reshape(face_val_data, (face_val_data.shape[0], face_val_data.shape[1], face_val_data.shape[2], 1))
face_test_data = np.reshape(face_test_data, (face_test_data.shape[0], face_test_data.shape[1], face_test_data.shape[2], 1))
eye_train_data = np.reshape(eye_train_data, (eye_train_data.shape[0], eye_train_data.shape[1], eye_train_data.shape[2], 1))
eye_val_data = np.reshape(eye_val_data, (eye_val_data.shape[0], eye_val_data.shape[1], eye_val_data.shape[2], 1))
eye_test_data = np.reshape(eye_test_data, (eye_test_data.shape[0], eye_test_data.shape[1], eye_test_data.shape[2], 1))
mouth_train_data = np.reshape(mouth_train_data, (mouth_train_data.shape[0], mouth_train_data.shape[1], mouth_train_data.shape[2], 1))
mouth_val_data = np.reshape(mouth_val_data, (mouth_val_data.shape[0], mouth_val_data.shape[1], mouth_val_data.shape[2], 1))
mouth_test_data = np.reshape(mouth_test_data, (mouth_test_data.shape[0], mouth_test_data.shape[1], mouth_test_data.shape[2], 1))

 # 构建CNN-LSTM模型
face_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Reshape((1, -1)),
    tf.keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=False),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7, activation='softmax')
])

eye_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Reshape((1, -1)),
    tf.keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=False),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7, activation='softmax')
])

mouth_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Reshape((1, -1)),
    tf.keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=False),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7, activation='softmax')
])

face_model.compile(loss="categorical_crossentropy", optimizer="RMSprop", metrics=['accuracy'])
eye_model.compile(loss="categorical_crossentropy", optimizer="RMSprop", metrics=['accuracy'])
mouth_model.compile(loss="categorical_crossentropy", optimizer="RMSprop", metrics=['accuracy'])
# 定义EarlyStopping回调函数
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, verbose=1)
# 训练模型
face_model.fit(face_train_data, face_train_labels, batch_size=32, epochs=100, verbose=1, validation_data=(face_val_data, face_val_labels), callbacks=[early_stop])
eye_model.fit(eye_train_data, eye_train_labels, batch_size=32, epochs=100, verbose=1, validation_data=(eye_val_data, eye_val_labels), callbacks=[early_stop])
mouth_model.fit(mouth_train_data, mouth_train_labels, batch_size=32, epochs=100, verbose=1, validation_data=(mouth_val_data, mouth_val_labels), callbacks=[early_stop])
# 评估模型
face_score = face_model.evaluate(face_test_data, face_test_labels)
eye_score = eye_model.evaluate(eye_test_data, eye_test_labels)
mouth_score = mouth_model.evaluate(mouth_test_data, mouth_test_labels)
print("人脸测试准确率：", face_score[1])
print("眼部测试准确率：", eye_score[1])
print("嘴部测试准确率：", mouth_score[1])

# 保存模型
if not os.path.exists("model"):
    os.makedirs("model")
face_model.save("model/face_model.h5")
mouth_model.save("model/mouth_model.h5")
eye_model.save("model/eye_model.h5")

# 提取训练集中的人脸、眼睛和嘴巴特征
train_face_features = face_model.predict(face_train_data)
train_eye_features = eye_model.predict(eye_train_data)
train_mouth_features = mouth_model.predict(mouth_train_data)

# 提取验证集中的人脸、眼睛和嘴巴特征
val_face_features = face_model.predict(face_val_data)
val_eye_features = eye_model.predict(eye_val_data)
val_mouth_features = mouth_model.predict(mouth_val_data)

# 提取测试集中的人脸、眼睛和嘴巴特征
test_face_features = face_model.predict(face_test_data)
test_eye_features = eye_model.predict(eye_test_data)
test_mouth_features = mouth_model.predict(mouth_test_data)

# 将三个特征拼接在一起作为新的输入特征
train_features = np.concatenate([
    np.multiply(train_face_features, 0.6),
    np.multiply(train_eye_features, 0.2),
    np.multiply(train_mouth_features, 0.2)
], axis=1)

val_features = np.concatenate([
    np.multiply(val_face_features, 0.6),
    np.multiply(val_eye_features, 0.2),
    np.multiply(val_mouth_features, 0.2)
], axis=1)

test_features = np.concatenate([
    np.multiply(test_face_features, 0.6),
    np.multiply(test_eye_features, 0.2),
    np.multiply(test_mouth_features, 0.2)
], axis=1)

# 构建新的模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7, activation='softmax')
])
model.compile(loss="categorical_crossentropy", optimizer="RMSprop", metrics=['accuracy'])
# 定义EarlyStopping回调函数
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, verbose=1)
# 训练模型
model.fit(train_features, face_train_labels, batch_size=32, epochs=100, verbose=1, validation_data=(val_features, face_val_labels), callbacks=[early_stop])
# 评估模型
score = model.evaluate(test_features, face_test_labels)
print("测试准确率：", score[1])
# 保存模型
if not os.path.exists("model"):
    os.makedirs("model")
model.save("model/Multi_Model.h5")
