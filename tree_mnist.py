# -*- coding:utf-8 -*-
"""
@Time: 2022/03/11 21:28
@Author: KI
@File: tree_mnist.py
@Motto: Hungry And Humble
"""
from sklearn.tree import DecisionTreeClassifier
from knn_mnist import load_data


def main():
    train_x, train_y, test_x, test_y = load_data()
    model = DecisionTreeClassifier()
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y)
    print('accuracy:', score)


if __name__ == '__main__':
    main()
