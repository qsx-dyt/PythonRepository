import pickle
import cv2
import numpy as np
import tensorflow as tf


def detect(data):
    # 加载模型
    model = tf.keras.models.load_model('model/VGG-LSTM.h5')

    # 加载SVM分类器
    with open('model/svm.pkl', 'rb') as f:
        clf = pickle.load(f)

    # 定义标签列表
    ckp_labels = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    fer_labels = ['anger', 'contempt', 'fear', 'happy', 'sadness', 'surprise', 'neutral']

    # 加载人脸检测器
    face_cascade = cv2.CascadeClassifier('util/haarcascade_frontalface_default.xml')

    # 初始化
    cap = cv2.VideoCapture(data)

    while True:
        # 读取一帧图像
        ret, frame = cap.read()
        if ret:
            # 转换为**图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 在图像中检测人脸
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

            # 遍历每个人脸并进行表情分类
            for (x, y, w, h) in faces:
                # 提取人脸区域并进行预处理
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray / 255.0
                roi_gray = roi_gray.reshape(1, 48, 48, 3)
                # 进行表情分类
                features = model.predict(roi_gray)
                features = features.reshape(features.shape[0], -1)
                # label = clf.predict(features)[0]
                label = np.argmax(features)
                print(label)
                # 在图像中显示表情分类结果
                cv2.putText(frame, ckp_labels[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return frame
            # 在窗口中显示图像
            cv2.imshow('Emotion Detection', frame)

        # 按下q键退出程序
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()
