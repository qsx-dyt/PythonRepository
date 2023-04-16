import cv2
import tensorflow as tf
import sys
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog
from PyQt5.uic import loadUi


class EmotionRecognitionUI(QDialog):
    def __init__(self):
        super().__init__()
        # 创建计时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        # UI设置
        self.capture = None
        loadUi('util/form.ui', self)
        self.cameraButton.clicked.connect(self.start_camera_recognition)
        self.imageButton.clicked.connect(self.select_image)
        self.videoButton.clicked.connect(self.start_video_recognition)
        self.stopButton.clicked.connect(self.stop_recognition)

        # 人脸分类器初始化
        self.face_cascade = cv2.CascadeClassifier('util/haarcascade_frontalface_default.xml')
        # 加载模型
        self.model = tf.keras.models.load_model('model/VGG-LSTM.h5')
        # 定义标签列表
        self.ckp_labels = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
        self.fer_labels = ['anger', 'contempt', 'fear', 'happy', 'sadness', 'surprise', 'neutral']

        """# 加载人脸检测器
        detector = cv2.dnn.readNetFromCaffe("util/deploy.prototxt.txt", "util/res10_300x300_ssd_iter_140000.caffemodel")"""

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
        self.timer.stop()
        self.capture.release()
        self.frame_label.clear()
        self.frame_label.clear()

    def emotion_recognition(self, frame):
        # 人脸检测
        faces = self.face_cascade.detectMultiScale(frame, 1.3, 5)
        for (x, y, w, h) in faces:
            # 人脸图像
            face = frame[y:y + h, x:x + w]
            # 调整图像大小为模型的输入大小
            face = cv2.resize(face, (48, 48))
            # 将图像转换为灰度图像
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            # 将图像标准化到 [0, 1] 的范围内
            image = gray.astype("float") / 255.0
            # 将图像转换为 4D 张量
            image = image.reshape(1, 48, 48, 3)
            # 将图像传递给模型进行分类
            preds = self.model.predict(image)
            label = self.ckp_labels[preds.argmax()]
            # 画矩形框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # 显示表情
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            # 人脸检测
            frame = self.emotion_recognition(frame)
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
            result = self.emotion_recognition(image)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            result = QImage(result.data, result.shape[1], result.shape[0], result.shape[1]*3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(result)
            self.frame_label.setPixmap(pixmap)
            self.frame_label.setScaledContents(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = EmotionRecognitionUI()
    ui.show()
    sys.exit(app.exec_())
