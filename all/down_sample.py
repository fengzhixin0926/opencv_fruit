import os
import numpy as np
from PIL import Image
import cv2
def fruit_resize(image):
    h, w = image.shape[0:2]
    img = Image.fromarray(image)
    background = Image.new('RGB', size=(max(image.shape[0:2]), max(image.shape[0:2])), color=(0,0,0))
    space = int(abs(w - h) // 2)
    box = (space, 0) if w < h else (0, space)
    background.paste(img, box)
    return np.asarray(background)

if __name__ == "__main__":
    path = 'F:/OpenCV/fruit/'
    fileList = os.listdir(path)
    for item in fileList:
        fruit = cv2.imread(path + item)
        # 转换为正方形
        new_fruit = fruit_resize(fruit)
        size = (64, 64)
        # 下采样
        new_img = cv2.resize(new_fruit, size)
        cv2.imwrite("F:/OpenCV/resize/" + item, new_img)
    print("Done!")

