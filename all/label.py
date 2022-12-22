import os
import numpy as np
import cv2
fruits = {'apple': 1, 'banana': 2, 'kiwifruit': 3, 'mango': 4, 'orange': 5,
          'passion': 6, 'pear': 7, 'pineapple': 8,  'strawberry': 9, 'unknow': 10, 'watermelon': 11}
data_dir = "F:/OpenCV/fruit_single/"
data_list = []  # 输入特征
labels = []  # 标签
for dir in os.listdir(data_dir):
    for file in os.listdir(data_dir + dir):   # 打开里面的子目录
        fruit = cv2.imread(data_dir + dir + "/" + file)
        # im = data_dir +file
        data_list.append(fruit)
        labels.append(fruits[dir] - 1)

state = np.random.get_state()
np.random.shuffle(data_list)
np.random.set_state(state)
np.random.shuffle(labels)
np.savez("F:/OpenCV/all/fruit.npz", data_list=np.array(data_list, dtype=object),
         labels=np.array(labels, dtype=object))

print("Done!")