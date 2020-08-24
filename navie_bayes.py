# encoding=utf-8

import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
import time


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def binaryzation(img):
    cv_img = img.astype(np.uint8)
    ret, cv_img = cv2.threshold(cv_img, 50, 1, cv2.THRESH_BINARY_INV)
    # cv2.threshold(cv_img, 127, 255, cv2.CV_THRESH_BINARY_INV)
    # cv2.threshold(cv_img, 50, 1, cv2.cv.CV_THRESH_BINARY_INV, cv_img)
    return cv_img

def Train(train_set, train_labels):
    prior_prob = np.zeros(class_num)  # (10,1)
    cond_prob = np.zeros((class_num, feature_len, 2)) # 784* 2 784行2列 10维

    for i in range(len(train_labels)):
        img = binaryzation(train_set[i])  # 二值化 img 中只有0 1
        label = train_labels[i]
        prior_prob[label] += 1

        for j in range(feature_len):
            cond_prob[label][j][img[j]] += 1

    for i in range(class_num):
        for j in range(feature_len):
            pix_0 = cond_prob[i][j][0] # pix_0 + pix_1 是所有label[i](即y=ci)下的和
            pix_1 = cond_prob[i][j][1]

            cond_prob_0 = (float(pix_0)/float(pix_0 + pix_1)) * 1000000 + 1  # p(x=aij|y=ci) = sum(x=aij|y=ci)/sum(y=ci)
            cond_prob_1 = (float(pix_1)/float(pix_0 + pix_1)) * 1000000 + 1

            cond_prob[i][j][0] = cond_prob_0
            cond_prob[i][j][1] = cond_prob_1
    return prior_prob, cond_prob

def cal_prob(img, label):
    probability = int(prior_prob[label])
    for i in range(len(img)):
        probability *= int(cond_prob[label][i][img[i]])
    return probability



def predict(test_set):  # 求argmax最大
    predict = []
    for img in test_set:
        img = binaryzation(img)
        max_level = 0
        max_prob = cal_prob(img, 0)

        for j in range(1,10):
            prob = cal_prob(img, j)
            if max_prob < prob:
                max_level = j
                max_prob = prob
        predict.append(max_level)
    return np.array(predict)




class_num = 10
feature_len = 784

# test ead image and rgb2gray
# img = cv2.imread('D:/faste_rcnn/00001.png')
# GrayImage = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# binary_img = binaryzation(GrayImage)
# ret, binary_img  = cv2.threshold(GrayImage, 50,1, cv2.THRESH_BINARY_INV)
# plt.imshow(binary_img,'gray')
# plt.show()



if __name__ == '__main__':
    time_1 = time.time()
    raw_data = pd.read_csv('data/train.csv', header=0)
    data = raw_data.values  # 取内容 不取title
    imgs = data[:,1:]
    labels = data[:,0]
    train_features, test_features, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.33, random_state=23323)
    print(train_features.shape, train_labels.shape)  # (28140, 784) (28140,)
    time_2 = time.time()
    prior_prob, cond_prob = Train(train_features, train_labels)

    time_3 = time.time()
    test_predict = predict(test_features)
    score = accuracy_score(test_labels, test_predict)
    print(score)
