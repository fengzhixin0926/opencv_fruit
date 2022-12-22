import cv2
import numpy as np
def pre(path, num):
    # mog2_sub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    knn_sub = cv2.createBackgroundSubtractorKNN(detectShadows=True)
    path = path + "%d.jpg"
    fruit_pic = [None] * (num+1)   # 600张
    k = 1
#   借鉴课本140页
    for i in range(1, num+1):
        output_name = "E:/kechengsheji/data/end/%d.jpg" % i
        fruit_pic[i] = cv2.imread(path % i)
        # print(fruit_pic[i])
        # mog2_sub_mask = mog2_sub.apply(fruit_pic[i])
        knn_sub_mask = knn_sub.apply(fruit_pic[i])
        # 矩形内核
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # th = cv2.threshold(mog2_sub_mask.copy(), 222, 255, cv2.THRESH_BINARY)[1]
        th = cv2.threshold(knn_sub_mask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
        eroded = cv2.erode(th, kernel, iterations=2)  # 腐蚀
        dilated = cv2.dilate(eroded, kernel, iterations=2)  # 膨胀
        cv2.medianBlur(dilated, 5)  # 中值滤波
        # print(len(cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)))  3
        _, contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 去除背景
        # no_background = fruit_pic[i] & mog2_sub_mask[:, :, np.newaxis]
        no_background = fruit_pic[i] & knn_sub_mask[:, :, np.newaxis]
        # cv2.imshow("mog2", no_background)
        cv2.imwrite(output_name, no_background)
        # for c in contours:
        #     (x, y, w, h) = cv2.boundingRect(c)
        #     fruit = fruit_pic[i][y:y+h, x:x+w]
        #     # cv2.imwrite("F:/OpenCV/fruit/%d.jpg" % k, fruit)
        #     k += 1
        #     if cv2.waitKey(10) & 0xff == 27:
        #         break
if __name__ == "__main__":
    path = "E:/kechengsheji/data/label/"
    num = 23
    pre(path, num)
    print("Done!")

# 此处报错：ValueError: too many values to unpack (expected 2)
# opencv2返回两个值：contours：hierarchy。opencv3会返回三个值,分别是img, countours, hierarchy
# 通过测试，发现222最好（222,200,244）

