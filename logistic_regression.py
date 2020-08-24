import time
import math
import random

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
class LogisticRegression(object):
    def __init__(self):
        self.learning_step = 0.00001
        self.max_iteration = 5000

    def predict_(self, x):
        wx = sum([self.w[i] * x[i] for i in range(len(self.w))])
        epx_wx = math.exp(wx)
        predict1 = epx_wx/(1 + epx_wx)
        predict0 = 1/(1 + epx_wx)
        if predict0 > predict1:
            return 0
        else:
            return 1


    def train(self, features, labels):  #train_features, train_labels
        self.w = [0.0] * len((features[0]) + 1)
        correct_count = 0
        time = 0
        while time < self.max_iteration:
            index = random.randint(0, len(labels) - 1)
            x = list(features[index])
            x.append(1.0)
            y = labels[index]
            if y == self.predict_(x):
                correct_count += 1
                if correct_count > self.max_iteration:
                    break
                continue

            time += 1
            correct_count = 0

            wx = sum([self.w[i] * x[i] for i in range(len(self.w))])
            exp_wx = math.exp(wx)

            for i in range(len(self.w)):
                self.w[i] -= self.learning_step * (-y * x[i] + float(x[i] * exp_wx) / float(1 + exp_wx))


    def predict(self, features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)
            labels.append(self.predict_(x))
        return labels


if __name__ == '__main__':
    print('start read data')

    time_1 = time.time()
    raw_data = pd.read_csv('data/train.csv', header=0)
    data = raw_data.values  # 取内容 不取title

    imgs = data[:, 1:]
    labels = data[:, 0]  # 42000
    train_features, test_features, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.33,
                                                                                random_state=23323)
    time_2 = time.time()
    print('read data cost', time_2 - time_1, 'second')
    print('start training')
    lr = LogisticRegression()
    lr.train(train_features, train_labels)
    test_predict = lr.predict(test_features)
    score = accuracy_score(test_labels, test_predict)