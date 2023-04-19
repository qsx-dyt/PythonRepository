import os.path
import h5py
import numpy as np
import tensorflow as tf

tf.config.list_physical_devices('GPU')

data_path = "dataset/gray/"

# 加载HDF5格式的数据集
with h5py.File(data_path+'ckp.hdf5', 'r') as f:
    train_data = f['train_data'][:]
    train_labels = f['train_labels'][:]
    val_data = f['val_data'][:]
    val_labels = f['val_labels'][:]
    test_data = f['test_data'][:]
    test_labels = f['test_labels'][:]

# 将数据集调整为适合CNN-LSTM模型的形状
train_data = np.reshape(train_data, (train_data.shape[0], 1, train_data.shape[1], train_data.shape[2], 1))
val_data = np.reshape(val_data, (val_data.shape[0], 1, val_data.shape[1], val_data.shape[2], 1))
test_data = np.reshape(test_data, (test_data.shape[0], 1, test_data.shape[1], test_data.shape[2], 1))

print(train_data.shape)
print(val_data.shape)
print(test_data.shape)

# 构建CNN-LSTM模型
model = tf.keras.Sequential([
    tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                                    input_shape=(1, 48, 48, 1)),
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
history = model.fit(train_data, train_labels, batch_size=32, epochs=100, verbose=1,
                    validation_data=(val_data, val_labels), callbacks=[tensorboard_callback, early_stop])

# 评估模型
score = model.evaluate(test_data, test_labels)
print("测试准确率：", score[1])

# 保存模型
if not os.path.exists("model"):
    os.makedirs("model")
model.save("model/CNN-LSTM_1.h5")
