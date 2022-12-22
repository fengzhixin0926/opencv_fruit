import os
import cv2
path = "F:/OpenCV/background/"
fileList = os.listdir(path)
for item in fileList:
    img = cv2.imread(path + item)
    b, g, r = cv2.split(img)
    cv2.imshow('b', b)
    cv2.imshow('g', g)
    cv2.imshow('r', r)
    cv2.imwrite("F:/OpenCV/b/" + item, b)
    cv2.imwrite("F:/OpenCV/g/" + item, g)
    cv2.imwrite("F:/OpenCV/r/" + item, r)

print("Done!")

