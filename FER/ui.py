import numpy as np
import cv2
import tensorflow as tf
import sys

from PyQt5 import QtCore
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog
from PyQt5.uic import loadUi
from mtcnn import MTCNN
from qt_material import apply_stylesheet


class EmotionRecognitionUI(QDialog):
    def __init__(self):
        super().__init__()
        self.model3 = None
        self.model2 = None
        self.model1 = None
        self.ismoving = None
        self.window_point = None
        self.start_point = None
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # 设置窗口背景透明
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)  # 隐藏边框
        self.capture = None
        # 人脸分类器初始化
        self.face_cascade = cv2.CascadeClassifier('util/haarcascade_frontalface_default.xml')
        self.detector = MTCNN()
        # 加载模型
        self.model = tf.keras.models.load_model('model/VGG-LSTM_1.h5')
        self.modelText = 'VGG-LSTM_1'
        # 定义标签列表
        self.ckp_labels = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
        self.fer_labels = ['anger', 'contempt', 'fear', 'happy', 'sadness', 'surprise', 'neutral']
        self.labels = self.ckp_labels
        # 创建计时器
        self.frame_count = 0  # 添加计数器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        # UI设置
        loadUi('util/form.ui', self)
        self.ModelcomboBox.currentTextChanged.connect(self.select_model)
        self.cameraButton.clicked.connect(self.start_camera_recognition)
        self.imageButton.clicked.connect(self.select_image)
        self.videoButton.clicked.connect(self.start_video_recognition)
        self.stopButton.clicked.connect(self.stop_recognition)
        self.closeButton.clicked.connect(self.close_window)
        self.miniButton.clicked.connect(self.minimize)

    def select_model(self, text):
        # 加载模型
        self.modelText = text
        self.model = tf.keras.models.load_model(f'model/{text}.h5')
        if self.modelText == 'Multi_Model':
            self.model1 = tf.keras.models.load_model(f'model/face_model.h5')
            self.model2 = tf.keras.models.load_model(f'model/mouth_model.h5')
            self.model3 = tf.keras.models.load_model(f'model/eye_model.h5')
        if text[-2:] == '_2':
            self.labels = self.fer_labels
        else:
            self.labels = self.ckp_labels

    def start_camera_recognition(self):
        # 打开摄像头进行实时表情识别
        self.capture = cv2.VideoCapture(0)
        self.timer.start(20)

    def start_video_recognition(self):
        # 选择视频进行实时表情识别
        file_name, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video Files (*.mp4 *.avi)")
        if file_name:
            self.capture = cv2.VideoCapture(file_name)
            self.timer.start(20)

    def stop_recognition(self):
        # 停止计时器和视频捕获
        if self.timer.isActive():
            self.timer.stop()
        if self.capture:
            self.capture.release()
        self.frame_label.clear()
        self.time.clear()
        self.probability.clear()
        self.result.clear()

    def close_window(self):
        self.close()

    def minimize(self):
        self.showMinimized()

    def mousePressEvent(self, e):
        self.start_point = e.globalPos()
        self.window_point = self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        self.ismoving = True
        relpos = e.globalPos() - self.start_point
        self.move(self.window_point + relpos)

    def mouseReleaseEvent(self, e):
        self.ismoving = False

    def emotion_recognition(self, frame):
        start_time = cv2.getTickCount()
        # 人脸检测
        # 将图像转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5)
        for (x, y, w, h) in faces:
            # 人脸图像
            face = frame[y:y + h, x:x + w]
            # 调整图像大小为模型的输入大小
            face = cv2.resize(face, (48, 48))

            if self.modelText[0:-2] == "CNN-LSTM":
                # 将图像转换为灰度图像
                image = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                # 将图像标准化到 [0, 1] 的范围内
                image = image.astype("float") / 255.0
                image = image.reshape(1, 1, 48, 48, 1)
            else:
                # 将图像转换为RGB图像
                image = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                # 将图像标准化到 [0, 1] 的范围内
                image = image.astype("float") / 255.0
                image = image.reshape(1, 48, 48, 3)
            # 将图像传递给模型进行分类
            preds = self.model.predict(image)
            label = self.ckp_labels[preds.argmax()]
            # 显示时间、结果
            end_time = cv2.getTickCount()
            time_in_sec = (end_time - start_time) / cv2.getTickFrequency()
            self.time.setText(str(time_in_sec))
            self.probability.setText(str(preds[0, preds.argmax()]))
            self.result.setText(label)
            # 画矩形框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # 显示表情
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

    def multiply_recognition(self, frame):
        start_time = cv2.getTickCount()
        # 人脸检测
        faces = self.detector.detect_faces(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for face in faces:
            x, y, w, h = face['box']
            # 人脸图像
            face_img = gray[y:y + h, x:x + w]
            # 调整图像大小为模型的输入大小
            face_img = cv2.resize(face_img, (48, 48))
            # 将图像标准化到 [0, 1] 的范围内
            face_img = face_img.astype("float") / 255.0
            face_img = face_img.reshape(1, 48, 48, 1)
            face_pred = self.model1.predict(face_img)
            # 嘴部
            mouth_x = face['keypoints']['mouth_left'][0]
            mouth_y = min(face['keypoints']['mouth_left'][1], face['keypoints']['mouth_right'][1])
            mouth_img = gray[mouth_y - 30:mouth_y + 40, mouth_x:face['keypoints']['mouth_right'][0]]
            mouth_img = cv2.resize(mouth_img, (48, 48))
            mouth_img = mouth_img.astype("float") / 255.0
            mouth_img = mouth_img.reshape(1, 48, 48, 1)
            mouth_pred = self.model2.predict(mouth_img)
            # 眼部
            eye_x = face['keypoints']['left_eye'][0]
            eye_y = min(face['keypoints']['left_eye'][1], face['keypoints']['right_eye'][1])
            eyes_img = gray[eye_y - 20:eye_y + 30, eye_x - 50:face['keypoints']['right_eye'][0] + 50]
            eyes_img = cv2.resize(eyes_img, (48, 48))
            eyes_img = eyes_img.astype("float") / 255.0
            eyes_img = eyes_img.reshape(1, 48, 48, 1)
            eyes_pred = self.model3.predict(eyes_img)
            # 融合
            data = np.concatenate([np.multiply(face_pred, 0.6), np.multiply(mouth_pred, 0.2), np.multiply(eyes_pred, 0.2)], axis=1)
            preds = self.model.predict(data)
            label = self.ckp_labels[preds.argmax()]
            # 显示时间、结果
            end_time = cv2.getTickCount()
            time_in_sec = (end_time - start_time) / cv2.getTickFrequency()
            self.time.setText(str(time_in_sec))
            self.probability.setText(str(preds[0, preds.argmax()]))
            self.result.setText(label)
            # 画矩形框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # 显示表情
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

    def update_frame(self):
        ret, frame = self.capture.read()
        self.frame_count += 1
        if ret and self.frame_count % 1 == 0:
            # 人脸检测
            if self.modelText == "Multi_Model":
                frame = self.multiply_recognition(frame)
            else:
                frame = self.emotion_recognition(frame)
        if ret:
            # 将视频帧显示在UI界面上
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(frame)
            self.frame_label.setPixmap(pixmap)
            self.frame_label.setScaledContents(True)

    def select_image(self):
        # 选择文件
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.jpg *.png)")
        if file_name:
            # 读取图像
            image = cv2.imread(file_name)
            # 人脸检测
            if self.modelText == "Multi_Model":
                result = self.multiply_recognition(image)
            else:
                result = self.emotion_recognition(image)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            result = QImage(result.data, result.shape[1], result.shape[0], result.shape[1]*3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(result)
            self.frame_label.setPixmap(pixmap)
            self.frame_label.setScaledContents(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    """# 加载样式表
    with open("util/style.qss", "r") as f:
        style = f.read()
        app.setStyleSheet(style)"""
    ui = EmotionRecognitionUI()
    # apply_stylesheet(app, theme='light_blue.xml')
    ui.show()
    sys.exit(app.exec_())
