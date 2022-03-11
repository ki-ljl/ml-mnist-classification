# -*- coding: utf-8 -*-
"""
@Time ： 2022/1/7 21:28
@Author ：KI 
@File ：knn_mnist.py
@Motto：Hungry And Humble

"""
import numpy as np
import torchvision
import torchvision.transforms as transforms
from sklearn.neighbors import KNeighborsClassifier


def load_data():
    dataset_train = torchvision.datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor())
    dataset_test = torchvision.datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())
    data_train = dataset_train.data
    X_train = data_train.numpy()
    X_test = dataset_test.data.numpy()
    X_train = np.reshape(X_train, (60000, 784))
    X_test = np.reshape(X_test, (10000, 784))
    Y_train = dataset_train.targets.numpy()
    Y_test = dataset_test.targets.numpy()

    return X_train, Y_train, X_test, Y_test


def get_distance(x1, x2):
    return np.linalg.norm(x1 - x2)


def get_vec(K, x, train_x, train_y):
    res = []
    for i in range(len(train_x)):
        dis = get_distance(x, train_x[i])
        res.append([dis, train_y[i]])

    res = sorted(res, key=(lambda t: t[0]))

    return res[:K]


def knn(K):
    train_x, train_y, test_x, test_y = load_data()
    # part of data
    train_x, train_y, test_x, test_y = train_x[:6000], train_y[:6000], test_x[:1000], test_y[:1000]
    cnt = 0
    for i in range(len(test_x)):
        print(i)
        x = test_x[i]
        y = test_y[i]
        vec = get_vec(K, x, train_x, train_y)
        weight = []
        sum_distance = 0.0
        for j in range(K):
            sum_distance += vec[j][0]
        for j in range(K):
            weight.append([1 - vec[j][0] / sum_distance, vec[j][1]])
        #
        num = []
        for j in range(K):
            num.append(weight[j][1])
        num = list(set(num))

        final_res = []
        for j in range(len(num)):
            res = 0.0
            for k in range(len(weight)):
                if weight[k][1] == num[j]:
                    res += weight[k][0]
            final_res.append([res, num[j]])

        final_res = sorted(final_res, key=(lambda e: e[0]), reverse=True)  # sort

        if y == final_res[0][1]:
            cnt = cnt + 1
        # print(y, final_res[0][1])

    print('accuracy:', cnt / len(test_x))


def knn_sklearn():
    train_x, train_y, test_x, test_y = load_data()
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(train_x, train_y)
    acc = knn.score(test_x, test_y)
    print('accuracy:', acc)


if __name__ == '__main__':
    K = 10
    knn(K)
    knn_sklearn()
