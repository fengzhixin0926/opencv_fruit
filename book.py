import cv2
import numpy as np
from random import randint

animals_net = cv2.ml.ANN_MLP_create()
animals_net.setTrainMethod(cv2.ml.ANN_MLP_RPROP | cv2.ml.ANN_MLP_UPDATE_WEIGHTS)
animals_net.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
animals_net.setLayerSizes(np.array(([3, 8, 4])))
# 指定ANN的终止条件
animals_net.setTermCriteria((cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))

"""Input arrays
weight, length, teeth
"""

"""one-hot
randint中的数值可以随便填
"""

"""Output arrays
dog, eagle, dolphin and dragon
(对每个类别的打分，以及最终的结果)

"""

def dog_sample():
    return [randint(5, 20), 1, randint(38, 42)]

def dog_class():
    return [1, 0, 0, 0]

def condor_sample():
    return [randint(3, 13), 3, 0]

def condor_class():
    return [0, 1, 0, 0]

def dolphin_sample():
    return [randint(30, 190), randint(5, 15), randint(80, 100)]

def dolphin_class():
    return [0, 0, 1, 0]

def dragon_sample():
    return [randint(1200, 1800), randint(15, 40), randint(110, 180)]

def dragon_class():
    return [0, 0, 0, 1]

SAMPLES = 5000
for x in range(0, SAMPLES):
    print("Samples %d/%d" % (x, SAMPLES))
    animals_net.train(np.array([dog_sample()], dtype=np.float32), cv2.ml.ROW_SAMPLE,
                      np.array([dog_class()], dtype=np.float32))
    animals_net.train(np.array([condor_sample()], dtype=np.float32), cv2.ml.ROW_SAMPLE,
                      np.array([condor_class()], dtype=np.float32))
    animals_net.train(np.array([dragon_sample()], dtype=np.float32), cv2.ml.ROW_SAMPLE,
                      np.array([dragon_class()], dtype=np.float32))
    animals_net.train(np.array([dolphin_sample()], dtype=np.float32), cv2.ml.ROW_SAMPLE,
                      np.array([dolphin_class()], dtype=np.float32))

print(animals_net.predict(np.array([dog_sample()], dtype=np.float32)))
print(animals_net.predict(np.array([condor_sample()], dtype=np.float32)))
print(animals_net.predict(np.array([dragon_sample()], dtype=np.float32)))
print(animals_net.predict(np.array([dolphin_sample()], dtype=np.float32)))


import numpy as np
import cv2
from matplotlib import pyplot as plt
img1 = cv2.imread('1', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('2', cv2.IMREAD_GRAYSCALE)
# 创建ORB特征检测器和描述符
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
# 暴力匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x:x.distance())
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:40], img2, flags=2)
plt.imshow(img3), plt.show()









