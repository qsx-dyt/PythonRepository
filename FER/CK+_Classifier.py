import os
import shutil

# 七种表情分类
emotions = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
# 表情文件
path = "CK+/cohn-kanade-images"
# 标签文件
path_label = "CK+/Emotion"

# 创建文件夹
if not os.path.exists('ClassifiedDataCK+/' + emotions[0]):
    for emotion in emotions:
        path_emotion = 'ClassifiedDataCK+' + '/' + emotion
        os.makedirs(path_emotion)

# 遍历所有文件夹
files = os.listdir(path)
for files_name in files:
    path_two = path + "/" + files_name
    path_two_label = path_label + "/" + files_name
    files_two = os.listdir(path_two)
    for files_two_name in files_two:
        if len(files_two_name) <= 4:
            path_three_label = os.path.join(path_two_label, files_two_name)
            if os.path.exists(path_three_label):
                path_three_label_file = os.listdir(path_three_label)
                path_three = os.path.join(path_two, files_two_name)
                path_three_file = os.listdir(path_three)
                print(path_three_file)
                if len(path_three_label_file) != 0:
                    f = open(os.path.join(path_three_label, path_three_label_file[0]))
                    lines = f.readline()
                    label = int(lines[3])
                    for i in range(8, len(path_three_file), 3):
                        name = os.path.join(path_three, path_three_file[i])
                        path_emotions = 'ClassifiedDataCK+' + '/' + emotions[label - 1]
                        shutil.copy(name, path_emotions)
