# encoding=utf-8

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

total_class = 10

def binaryzation(img):
    cv_img = img.astype(np.uint8)
    ret, cv_img = cv2.threshold(cv_img, 50, 1, cv2.THRESH_BINARY_INV)
    # cv2.threshold(cv_img, 127, 255, cv2.CV_THRESH_BINARY_INV)
    # cv2.threshold(cv_img, 50, 1, cv2.cv.CV_THRESH_BINARY_INV, cv_img)
    return cv_img

def binaryzation_features(train_set):
    features = []
    for img in train_set:
        img = np.reshape(img, (28,28))
        img = img.astype(np.uint8)
        cv_img = binaryzation(img)
        features.append(cv_img)

    features = np.array(features)
    features = np.reshape(features, (-1,784))
    return features

class Tree(object):
    def __init__(self, node_type, Class = None, feature = None):
        self.node_type = node_type
        self.Class = Class
        self.feature = feature
        self.dict = {}
    def add_tree(self, val, tree):
        self.dict[val] = tree

    def predict(self, feature):
        if self.node_type == 'leaf':
            return self.Class
        tree = self.dict[features[self.feature]]
        return tree.predict(features)

def cal_entro(x): # 信息熵/经验熵
    '''
    calculate entropy H(x)
    :param x: labels
    :return: H(D)
    '''
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0])/x.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    return ent

def cal_condition_ent(x,y):  # 条件熵
    """

    :param x: x-features 0~784
    :param y:- train_labels
    :return: H(Y|X)
    """
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        sub_y = y[x == x_value]  # for i in len(x): if x[i] == x_value sub_y.append(y[i])
        tmp_ent = cal_entro(sub_y)  # P(X=xi) euqals to sub_y.shape[0]/y.shape[0]
        ent += (float(sub_y.shape[0])/y.shape[0]) * tmp_ent
    return ent

def cal_ent_gda(x,y):
    base_ent = cal_entro(y)
    condition_ent = cal_condition_ent(x,y)
    gda_ent = base_ent - condition_ent
    return gda_ent




def recurse_train(train_set, train_labels, features, epsilon):
    global total_class
    LEAF = 'leaf'
    INTERNAL = 'internal'
    label_set = set(train_labels)

    # 如果只有一类
    if label_set == 1:
        return Tree(LEAF, Class=label_set.pop())
    (max_class, max_len) = max([(i, len(list(filter(lambda x: x == i, train_labels)))) for i in range(total_class)],
                               key=lambda x: x[1])


    if len(list(features)) == 0:
        return Tree(LEAF, Class=max_class)
    max_feature = 0
    max_gda = 0
    D = train_labels
    HD = cal_entro(D)
    for feature in features:
        A = np.array(train_set[:,feature].flat)
        gda = HD - cal_condition_ent(A,D)
        if gda > max_gda:
            max_gda, max_feature = gda, feature

    if max_gda < epsilon:
        return Tree(LEAF, Class=max_class)

    # 构建除了max_feature 之外的非空子集
    sub_features = filter(lambda x: x!=max_feature, features)
    tree = Tree(INTERNAL, feature=max_feature)
    feature_col = np.array(train_set[:, max_feature].flat)
    feature_value_list = set([feature_col[i] for i in range(feature_col.shape[0])])
    for feature_value in feature_value_list:
        index = []
        for i in range(len(train_labels)):
            if train_set[i][max_feature] == feature_value:
                index.append(i)
        sub_train_set = train_set[index]
        sub_train_label = train_labels[index]
        sub_tree = recurse_train(sub_train_set, sub_train_label, sub_features, epsilon)
        tree.add_tree(feature_value, sub_tree)
    return tree


def train(train_set,train_label,features,epsilon):
    return recurse_train(train_set,train_label,features,epsilon)


def predict(test_set,tree):

    result = []
    for features in test_set:
        tmp_predict = tree.predict(features)
        result.append(tmp_predict)
    return np.array(result)









if __name__ == '__main__':
    raw_data = pd.read_csv('data/train.csv', header=0)
    data = raw_data.values  # 取内容 不取title
    imgs = data[:, 1:]
    labels = data[:, 0] #42000
    features = binaryzation_features(imgs)  #42000*784
    train_features, test_features, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.33,
                                                                                random_state=23323)

    # train_features  28140*784
    # test_features   13860*784
    # train_labels   1*28140
    # test_labels   1*13860
    tree = train(train_features, train_labels, [i for i in range(784)], 0.1)  # train_set,train_label,features,epsilon
    test_predict = predict(test_features,tree)
    score = accuracy_score(test_labels, test_predict)
    print(score)

