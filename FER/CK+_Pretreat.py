import os
import cv2
import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 定义数据集路径、处理后图片保存路径、数据集路径和标签映射
dataset_path = "ClassifiedDataCK+"
image_save_path = "PretreatedDataCK+"
dataset_save_path = "dataset/color/"
emotion_labels = {"anger": 0, "contempt": 1, "disgust": 2, "fear": 3, "happy": 4, "sadness": 5, "surprise": 6}

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier("util/haarcascade_frontalface_default.xml")


# 读取数据集并进行预处理
def load_data():
    images = []
    labels = []
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        for filename in os.listdir(label_path):
            image_path = os.path.join(label_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            # 数据增强并保存
            for i in range(5):
                augmented_image = augment_data(image)
                faces = face_cascade.detectMultiScale(augmented_image, scaleFactor=1.1, minNeighbors=5)
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    augmented_image = augmented_image[y:y + h, x:x + w]
                    augmented_image = cv2.resize(augmented_image, (48, 48))
                    augmented_image = augmented_image.astype(np.float32) / 255.0
                    if not os.path.exists(os.path.join(image_save_path, label)):
                        os.makedirs(os.path.join(image_save_path, label))
                    save_image_path = os.path.join(image_save_path, label, str(i) + '-' + filename)
                    cv2.imwrite(save_image_path, augmented_image * 255)
                    images.append(augmented_image)
                    labels.append(emotion_labels[label])
            # 裁剪图像人脸部分
            faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                image = image[y:y + h, x:x + w]
                # 缩放
                image = cv2.resize(image, (48, 48))
                # 归一化
                image = image.astype(np.float32) / 255.0
                # 保存图像
                save_image_path = os.path.join(image_save_path, label, filename)
                cv2.imwrite(save_image_path, image * 255)
                images.append(image)
                labels.append(emotion_labels[label])
    return np.array(images), np.array(labels)


# 定义数据增强函数
def augment_data(image):
    # 随机旋转
    angle = np.random.randint(-15, 15)
    rows, cols, _ = image.shape
    m = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    image = cv2.warpAffine(image, m, (cols, rows))
    # 随机平移
    tx = np.random.randint(-10, 10)
    ty = np.random.randint(-10, 10)
    m = np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, m, (cols, rows))
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
    # 划分为训练集和测试集（8:2）
    train_data, test_data, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=None)
    # 划分测试集为测试集和验证集（5:5）
    test_data, val_data, test_labels, val_labels = train_test_split(test_data, test_labels, test_size=0.5, random_state=None)
    return train_data, train_labels, val_data, val_labels, test_data, test_labels


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
with h5py.File(dataset_save_path+'ckp.hdf5', 'w') as f:
    f.create_dataset('train_data', data=train_data)
    f.create_dataset('train_labels', data=train_labels)
    f.create_dataset('val_data', data=val_data)
    f.create_dataset('val_labels', data=val_labels)
    f.create_dataset('test_data', data=test_data)
    f.create_dataset('test_labels', data=test_labels)
