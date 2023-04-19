import h5py
import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf

# 定义路径
dataset_save_path = "dataset/color/"
path = "fer2013/"
dataset_path = image_path = path+"images/"

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier('util/haarcascade_frontalface_default.xml')

# 将fer2013.csv分类保存为图像文件，0=生气，1=厌恶，1=恐惧，3=开心，4=悲伤，5=惊讶，6=中性


def classify_images():
    # 加载fer2013数据集csv文件
    data = pd.read_csv(path + "fer2013.csv")

    # 将像素字符串转换为像素数组
    data['pixels'] = data['pixels'].apply(lambda x: np.array(x.split(' ')).astype('float32'))

    # 创建图像目录
    if not os.path.exists(path + 'images'):
        os.makedirs(path + 'images')
    for emotion in set(data['emotion']):
        emotion = str(emotion)
        if not os.path.exists(image_path + emotion):
            os.makedirs(image_path + emotion)

    # 将每个样本保存为图像文件
    for i in range(len(data)):
        image = data['pixels'][i].reshape((48, 48)).astype('uint8')
        emotion = data['emotion'][i]
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=3)
        if len(faces) > 0:
            filename = os.path.join(str(emotion), f"{i}.jpg")
            cv2.imwrite(image_path + filename, image)
            for j in range(3):
                image2 = augment_data(image)
                image2 = cv2.resize(image2, (48, 48))
                filename2 = os.path.join(str(emotion), f"{j}-{i}.jpg")
                cv2.imwrite(image_path + filename2, image2)


# 读取数据集并进行预处理
def load_data():
    images = []
    labels = []
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        for filename in os.listdir(label_path):
            images_path = os.path.join(label_path, filename)
            image = cv2.imread(images_path, cv2.IMREAD_COLOR)
            # 归一化
            image = image.astype(np.float32) / 255.0
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)


# 定义数据增强函数
def augment_data(image):
    # 随机缩放
    scale = np.random.uniform(0.8, 1.2)
    image = cv2.resize(image, None, fx=scale, fy=scale)
    # 随机水平翻转
    image = cv2.flip(image, 1)
    # 随机调整亮度、对比度、饱和度
    alpha = np.random.uniform(0.7, 1.3)  # 对比度调整系数
    beta = np.random.randint(-30, 30)  # 亮度调整值
    # 对比度和亮度调整
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return image


# 划分数据集
def split_data(images, labels):
    # 划分为训练集和测试集（7:3）
    train_data, test_data, train_labels, test_labels = train_test_split(images, labels, test_size=0.3, random_state=None)
    # 划分测试集为测试集和验证集（5:5）
    test_data, val_data, test_labels, val_labels = train_test_split(test_data, test_labels, test_size=0.5, random_state=None)
    return train_data, train_labels, val_data, val_labels, test_data, test_labels


# 分类
# classify_images()
# 加载数据集并进行预处理和划分
images, labels = load_data()
train_data, train_labels, val_data, val_labels, test_data, test_labels = split_data(images, labels)

# 打印训练集、验证集和测试集形状
print(f"训练集数据：{train_data.shape}，标签：{train_labels.shape}")
print(f"验证集数据：{val_data.shape}，标签：{val_labels.shape}")
print(f"测试集数据：{test_data.shape}，标签：{test_labels.shape}")

# 将标签转换为独热编码
train_labels = tf.keras.utils.to_categorical(train_labels)
val_labels = tf.keras.utils.to_categorical(val_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# 创建HDF5文件并写入数据集
with h5py.File(dataset_save_path+'fer2013.hdf5', 'w') as f:
    f.create_dataset('train_data', data=train_data)
    f.create_dataset('train_labels', data=train_labels)
    f.create_dataset('val_data', data=val_data)
    f.create_dataset('val_labels', data=val_labels)
    f.create_dataset('test_data', data=test_data)
    f.create_dataset('test_labels', data=test_labels)
