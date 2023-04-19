import os
import h5py
import tensorflow as tf

tf.config.list_physical_devices('GPU')

data_path = "dataset/color/"

# 加载HDF5格式的数据集
with h5py.File(data_path+'ckp.hdf5', 'r') as f:
    train_data = f['train_data'][:]
    train_labels = f['train_labels'][:]
    val_data = f['val_data'][:]
    val_labels = f['val_labels'][:]
    test_data = f['test_data'][:]
    test_labels = f['test_labels'][:]

vgg_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
model = tf.keras.Sequential([
    vgg_model,
    tf.keras.layers.Reshape(target_shape=(1, -1)),
    tf.keras.layers.LSTM(units=256, dropout=0.2, recurrent_dropout=0.2,
                         kernel_regularizer=tf.keras.regularizers.l2(0.01), return_sequences=False),
    tf.keras.layers.Dense(units=7, activation='softmax')
])
model.summary()

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# TensorBoard回调函数
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="log/VL", histogram_freq=1)

# 定义EarlyStopping回调函数
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# 训练模型
history = model.fit(train_data, train_labels, batch_size=32, epochs=100, verbose=1,
                    validation_data=(val_data, val_labels), callbacks=[tensorboard_callback, early_stop])

# 评估模型
score = model.evaluate(test_data, test_labels)
print("测试准确率：", score[1])

# 保存模型
if not os.path.exists("model"):
    os.makedirs("model")
model.save("model/VGG-LSTM_1.h5")
