import cv2
import numpy as np

def backcut(img):
    # 拷贝原始图像的副本作为备份
    imgCopy = img.copy()
    # 创建掩码蒙版
    mask = np.zeros(img.shape[:2], np.uint8)
   # 创建临时数组
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
   # 选择ROI
    rect = cv2.selectROI('img', img, False, False)
    x, y, w, h = rect   # 可以用于提取感兴趣的区域
   # 使用红色线条，在原始图像拷贝副本 imgCopy 上绘制 ROI 区域的矩形边框：
    cv2.rectangle(imgCopy, (x, y), (x + w, y + h), (0, 0, 255), 3)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    output = img*mask2[:, :, np.newaxis]
    cv2.destroyWindow('img')
    return output

def showRGB(input):
    # 输入图片为BGR格式
    b, g, r = cv2.split(input)
    cv2.imshow('b', b)
    cv2.imshow('g', g)
    cv2.imshow('r', r)
    # cv2.waitKey()

def Down_and_Media(img):
    # 目标区域下采样
    img_down = cv2.pyrDown(img)
    # 滤波核越大，图像越模糊
    output = cv2.medianBlur(img_down, 3)
    return output

if __name__ == '__main__':
    img = cv2.imread('F:/OpenCV/img1/img/20.jpg')

    '''调整图片的大小，输入图片大小相同'''
    img = cv2.resize(img, (600, 400))

    '''背景去除， 输入:img, 输出:img_backcut'''
    num = 0
    img_backcut = backcut(img)
    while (num!=-1):
        try:
            if num == 0:
                num = int(input('不进行区域截取则输入-1,且不可为0:'))
            else:
                img_backcut += backcut(img)
                num = int(input('不进行区域截取则输入-1,且不可为0:'))
        except EOFError:
            break

    '''RGB颜色空间显示,输入img_backcut, 无输出'''
    showRGB(img_backcut)

    '''目标区域下采样、中值滤波'''
    dst_img = Down_and_Media(img_backcut)

    cv2.imshow("result_result", dst_img)
    cv2.waitKey()




   # img = back - input
   # img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
   # # 低于lower_red和高于upper_red的部分分别变成0，lower_red～upper_red之间的值变成255
   # lower_red = np.array([0, 0, 0])
   # upper_red = np.array([255, 195, 255])
   # mask = cv2.inRange(img_HSV, lower_red, upper_red)  # lower20===>0,upper200==>0，lower～upper==>255
   # return mask