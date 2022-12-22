import pyautogui
import cv2
import numpy as np
import time

# 获取游戏的目标区域
def get_range():
    # 获得屏幕尺寸
    w0, h0 = pyautogui.size()
    print(str(w0) + ' ' + str(h0))
    print('10s后请将鼠标放在游戏范围的左上角')
    time.sleep(10)
    x1, y1 = pyautogui.position()
    print(str(x1) + ' ' + str(y1))
    print('10s后请将鼠标放在游戏范围的右下角')
    time.sleep(10)
    x2, y2 = pyautogui.position()
    print(str(x2) + ' ' + str(y2))
    print('x:' + str(x1) + '\n' + 'y:' + str(y1) + '\n' + 'w:' + str((x2 - x1)) + '\n' + 'h:' + str((y2 - y1)))
    return x1, y1, (x2 - x1), (y2 - y1)

def screenshot(x0, y0, w0, h0):

    # i 为第几张截图
    for i in range(1200):
        img = pyautogui.screenshot(region=[x0, y0, w0, h0])
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(r'img1/' + str(i) + r'.jpg', img)  # 该文件同级目录创建一个img的文件夹
        time.sleep(0.4)      # 每隔0.4s截一张图
        print("第{}张图片截图完成".format(i))

if __name__ == '__main__':
    # 获取游戏区域
    x, y, w, h = get_range()
    # 对游戏区域进行截图
    screenshot(x, y, w, h)