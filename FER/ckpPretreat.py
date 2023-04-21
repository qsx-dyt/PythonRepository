import os
import cv2
import h5py
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
from sklearn.model_selection import train_test_split

# 定义数据集路径、处理后图片保存路径、数据集路径和标签映射
dataset_path = "ClassifiedDataCK+"
images_path = "ckp"
dataset_save_path = "dataset/gray/"
emotion_labels = {"anger": 0, "contempt": 1, "disgust": 2, "fear": 3, "happy": 4, "sadness": 5, "surprise": 6}

# 加载 MTCNN 模型
detector = MTCNN()


# 读取数据集并进行预处理
def load_data():
    face_images = []
    face_labels = []
    eye_images = []
    eye_labels = []
    mouth_images = []
    mouth_labels = []
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        for filename in os.listdir(label_path):
            image_path = os.path.join(label_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # 数据增强并保存
            for i in range(6):
                augmented_image = augment_data(image)
                augmented_rgb = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
                # augmented_rgb = augmented_image
                face = detector.detect_faces(augmented_rgb)
                if len(face) > 0:
                    # 获取人脸区域并保存
                    x, y, w, h = face[0]['box']
                    face_img = augmented_image[y:y + h, x:x + w]
                    face_img2 = cv2.resize(face_img, (192, 192))
                    face_img2 = face_img2.astype(np.float32) / 255.0
                    faces_path = os.path.join(images_path, "faces")
                    if not os.path.exists(os.path.join(faces_path, label)):
                        os.makedirs(os.path.join(faces_path, label))
                    save_image_path = os.path.join(faces_path, label, str(i) + '-' + filename)
                    cv2.imwrite(save_image_path, face_img2 * 255)
                    face_images.append(face_img2)
                    face_labels.append(emotion_labels[label])
                    # 获取嘴部区域
                    mouth_x1, mouth_y1 = face[0]['keypoints']['mouth_left'][0], face[0]['keypoints']['mouth_left'][1]
                    mouth_x2, mouth_y2 = face[0]['keypoints']['mouth_right'][0], face[0]['keypoints']['mouth_right'][1]
                    mouth_x_min, mouth_x_max = min(mouth_x1, mouth_x2), max(mouth_x1, mouth_x2)
                    mouth_y_min, mouth_y_max = min(mouth_y1, mouth_y2), max(mouth_y1, mouth_y2)
                    mouth_img = augmented_image[y + mouth_y_min:y + mouth_y_max, x + mouth_x_min:x + mouth_x_max]
                    if mouth_img:
                        mouth_img = cv2.resize(mouth_img, (48, 48))
                        mouth_img = mouth_img.astype(np.float32) / 255.0
                        mouths_path = os.path.join(images_path, "mouths")
                        if not os.path.exists(os.path.join(mouths_path, label)):
                            os.makedirs(os.path.join(mouths_path, label))
                        save_image_path = os.path.join(mouths_path, label, str(i) + '-' + filename)
                        cv2.imwrite(save_image_path, mouth_img * 255)
                        mouth_images.append(mouth_img)
                        mouth_labels.append(emotion_labels[label])
                    # 获取眼部区域
                    eyes_x1, eyes_y1 = face[0]['keypoints']['left_eye'][0], face[0]['keypoints']['left_eye'][1]
                    eyes_x2, eyes_y2 = face[0]['keypoints']['right_eye'][0], face[0]['keypoints']['right_eye'][1]
                    eyes_x_min, eyes_x_max = min(eyes_x1, eyes_x2), max(eyes_x1, eyes_x2)
                    eyes_y_min, eyes_y_max = min(eyes_y1, eyes_y2), max(eyes_y1, eyes_y2)
                    eyes_img = augmented_image[y + eyes_y_min:y + eyes_y_max, x + eyes_x_min:x + eyes_x_max]
                    if eyes_img:
                        eyes_img = cv2.resize(eyes_img, (48, 48))
                        eyes_img = eyes_img.astype(np.float32) / 255.0
                        eyes_path = os.path.join(images_path, "eyes")
                        if not os.path.exists(os.path.join(eyes_path, label)):
                            os.makedirs(os.path.join(eyes_path, label))
                        save_image_path = os.path.join(eyes_path, label, str(i) + '-' + filename)
                        cv2.imwrite(save_image_path, eyes_img * 255)
                        eye_images.append(eyes_img)
                        eye_labels.append(emotion_labels[label])
    return np.array(face_images), np.array(face_labels), np.array(eye_images), np.array(eye_labels), np.array(mouth_images), np.array(mouth_labels)


# 定义数据增强函数
def augment_data(image):
    # 随机旋转
    angle = np.random.randint(-15, 15)
    rows, cols = image.shape
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
    # 随机调整亮度、对比度
    alpha = np.random.uniform(0.7, 1.3)  # 对比度调整系数
    beta = np.random.randint(-30, 30)  # 亮度调整值
    # 对比度和亮度调整
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return image


# 划分数据集
def split_data(images, labels):
    # 划分为训练集和测试集（8:2）
    train_data, test_data, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
    # 划分测试集为测试集和验证集（5:5）
    test_data, val_data, test_labels, val_labels = train_test_split(test_data, test_labels, test_size=0.5, random_state=42)
    return train_data, train_labels, val_data, val_labels, test_data, test_labels


# 加载数据集并进行预处理和划分
face_images, face_labels, eye_images, eye_labels, mouth_images, mouth_labels = load_data()
face_train_data, face_train_labels, face_val_data, face_val_labels, face_test_data, face_test_labels = split_data(face_images, face_labels)
eye_train_data, eye_train_labels, eye_val_data, eye_val_labels, eye_test_data, eye_test_labels = split_data(eye_images, eye_labels)
mouth_train_data, mouth_train_labels, mouth_val_data, mouth_val_labels, mouth_test_data, mouth_test_labels = split_data(mouth_images, mouth_labels)

# 打印训练集、验证集和测试集形状
print("faces:")
print(f"训练集数据：{face_train_data.shape}，标签：{face_train_labels.shape}\n"
      f"验证集数据：{face_val_data.shape}，标签：{face_val_labels.shape}\n"
      f"测试集数据：{face_test_data.shape}，标签：{face_test_labels.shape}")
print("eyes:")
print(f"训练集数据：{eye_train_data.shape}，标签：{eye_train_labels.shape}\n"
      f"验证集数据：{eye_val_data.shape}，标签：{eye_val_labels.shape}\n"
      f"测试集数据：{eye_test_data.shape}，标签：{eye_test_labels.shape}")
print("mouths:")
print(f"训练集数据：{mouth_train_data.shape}，标签：{mouth_train_labels.shape}\n"
      f"验证集数据：{mouth_val_data.shape}，标签：{mouth_val_labels.shape}\n"
      f"测试集数据：{mouth_test_data.shape}，标签：{mouth_test_labels.shape}")

"""# 将标签转换为独热编码
face_train_labels = tf.keras.utils.to_categorical(face_train_labels)
face_val_labels = tf.keras.utils.to_categorical(face_val_labels)
face_test_labels = tf.keras.utils.to_categorical(face_test_labels)
eye_train_labels = tf.keras.utils.to_categorical(eye_train_labels)
eye_val_labels = tf.keras.utils.to_categorical(eye_val_labels)
eye_test_labels = tf.keras.utils.to_categorical(eye_test_labels)
mouth_train_labels = tf.keras.utils.to_categorical(mouth_train_labels)
mouth_val_labels = tf.keras.utils.to_categorical(mouth_val_labels)
mouth_test_labels = tf.keras.utils.to_categorical(mouth_test_labels)

# 创建HDF5文件并写入数据集
with h5py.File(dataset_save_path + 'face_ckp.hdf5', 'w') as f:
    f.create_dataset('face_train_data', data=face_train_data)
    f.create_dataset('face_train_labels', data=face_train_labels)
    f.create_dataset('face_val_data', data=face_val_data)
    f.create_dataset('face_val_labels', data=face_val_labels)
    f.create_dataset('face_test_data', data=face_test_data)
    f.create_dataset('face_test_labels', data=face_test_labels)
with h5py.File(dataset_save_path + 'eye_ckp.hdf5', 'w') as f:
    f.create_dataset('eye_train_data', data=eye_train_data)
    f.create_dataset('eye_train_labels', data=eye_train_labels)
    f.create_dataset('eye_val_data', data=eye_val_data)
    f.create_dataset('eye_val_labels', data=eye_val_labels)
    f.create_dataset('eye_test_data', data=eye_test_data)
    f.create_dataset('eye_test_labels', data=eye_test_labels)
with h5py.File(dataset_save_path + 'mouth_ckp.hdf5', 'w') as f:
    f.create_dataset('mouth_train_data', data=mouth_train_data)
    f.create_dataset('mouth_train_labels', data=mouth_train_labels)
    f.create_dataset('mouth_val_data', data=mouth_val_data)
    f.create_dataset('mouth_val_labels', data=mouth_val_labels)
    f.create_dataset('mouth_test_data', data=mouth_test_data)
    f.create_dataset('mouth_test_labels', data=mouth_test_labels)
"""