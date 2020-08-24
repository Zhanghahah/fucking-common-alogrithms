import math
import random
import numpy as np
from sklearn import  datasets, cluster
import matplotlib.pyplot as plt

"""
centers: 初始化的cluster个数

"""

iris = datasets.load_iris()
gt = iris['target']
# print(gt)

data = iris['data'][:,:2]
x = data[:, 0]
y = data[:, 1]

class MyKmeans:
    def __init__(self, k, n=20):
        self.k = k
        self.n = n
    def fit(self, x, centers=None):
        if centers is None:
            idx = np.random.randint(low=0, high=len(x),size=self.k)
            centers = x[idx]
        inters = 0
        while inters < self.n:
            points_set = {key: [] for key in range(self.k)}  #  是个dict
            for p in x:
                nearest_index = np.argmin(np.sum((centers - p) ** 2, axis=1)**0.5)
                points_set[nearest_index].append(p)

            for idx in range(self.k):
                centers[idx] = sum(points_set[idx])/ len(points_set[idx])  # 求平均
            inters += 1
        return points_set, centers

my_kmeans = MyKmeans(3)  #k=3
points_set, centers = my_kmeans.fit(data)
print(points_set, centers)
