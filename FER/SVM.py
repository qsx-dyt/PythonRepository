# 将输出输入到SVM中进行表情分类
import pickle

import h5py
from sklearn import svm
import tensorflow as tf

data_path = "dataset/gray/"

# 加载CNN-LSTM模型
model = tf.keras.models.load_model('model/CNN-LSTM.h5')

# 加载HDF5格式的数据集
with h5py.File(data_path+'ckp.hdf5', 'r') as f:
    train_data = f['train_data'][:]
    train_labels = f['train_labels'][:]
    val_data = f['val_data'][:]
    val_labels = f['val_labels'][:]
    test_data = f['test_data'][:]
    test_labels = f['test_labels'][:]

# 将训练集、验证集和测试集输入到CNN-LSTM模型中，获取模型的输出
train_features = model.predict(train_data)
val_features = model.predict(val_data)
test_features = model.predict(test_data)

# 使用SVM分类器进行表情分类
clf = svm.SVC(kernel='linear')
clf.fit(train_features.reshape(train_features.shape[0], -1), train_labels.argmax(axis=1))
train_score = clf.score(train_features.reshape(train_features.shape[0], -1), train_labels.argmax(axis=1))
val_score = clf.score(val_features.reshape(val_features.shape[0], -1), val_labels.argmax(axis=1))
test_score = clf.score(test_features.reshape(test_features.shape[0], -1), test_labels.argmax(axis=1))

# 输出分类器在训练集、验证集和测试集上的表现
print('Train accuracy:', train_score)
print('Validation accuracy:', val_score)
print('Test accuracy:', test_score)

# 保存SVM分类器
with open('model/svm.pkl', 'wb') as f:
    pickle.dump(clf, f)
