import os
import h5py
import tensorflow as tf
import numpy as np

data_path = "dataset/gray/"

# 加载HDF5格式的数据集
with h5py.File(data_path+'ckp.hdf5', 'r') as f:
    train_data = f['train_data'][:]
    train_labels = f['train_labels'][:]
    val_data = f['val_data'][:]
    val_labels = f['val_labels'][:]
    test_data = f['test_data'][:]
    test_labels = f['test_labels'][:]

train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], train_data.shape[2], 1))
val_data = np.reshape(val_data, (val_data.shape[0], val_data.shape[1], val_data.shape[2], 1))
test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], test_data.shape[2], 1))

# 定义模型
model = tf.keras.models.Sequential([
    # AlexNet部分
    tf.keras.layers.InputLayer(input_shape=(48, 48, 1)),
    tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), strides=(2, 2), activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    tf.keras.layers.Flatten(),
    # LSTM
    tf.keras.layers.Reshape((1, 1024)),
    tf.keras.layers.LSTM(units=128, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=tf.keras.regularizers.l2(0.01), return_sequences=True),
    tf.keras.layers.LSTM(units=128, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(units=7, activation='softmax')
])

model.summary()

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# TensorBoard回调函数
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="log/AL", histogram_freq=1)

# 定义EarlyStopping回调函数
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1)

# 训练模型
history = model.fit(train_data, train_labels, batch_size=32, epochs=100, verbose=1,
                    validation_data=(val_data, val_labels), callbacks=[tensorboard_callback, early_stop])

# 评估模型
score = model.evaluate(test_data, test_labels)
print("测试准确率：", score[1])

# 保存模型
if not os.path.exists("model"):
    os.makedirs("model")
model.save("model/AlexNet-LSTM.h5")
