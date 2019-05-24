"""
一个多元线性回归的训练模型
利用了梯度下降算法，具体公式参考梯度下降算法
grad: 表示梯度
theta: 表示下降后每一次的位置， 一般初始位置设置为0
cost: 表示损失函数，就是与真实值的差距，一般设置为一个二元函数
feature: 特征值, 就是要参与训练的值，简单来说就是利用这些值来拟合出曲线，对应数学公式的x
label: 标签， 就是要预测的那条数据，就是实际值，对应数学公式中的y
learn_rate: 学习率，就是步长，每次下降的频度
"""

import numpy as np
from util import deal_data


def _get_grad(theta, feature, label):
    """
    梯度计算,利用了偏导
    由于使用了矩阵，就需要对feature进行转置，因为数学公式中引入矩阵后推导后的结果就是这样
    """
    grad = np.dot(np.transpose(feature), (np.dot(feature, theta) - label))
    return grad


def _get_theta(theta, feature, label, learn_rate):
    """
    计算下一theta
    按照公式这里应该除len(feature)，但考虑到计算机计算除法比较慢，这里只是一种思想，所以不加也可以
    """
    return theta - _get_grad(theta, feature, label, ) * learn_rate


def _get_cost(theta, feature, label):
    """
    误差函数，衡量拟合函数的效果
    误差会逐步逼近0
    一般训练数据与验证数据算出的曲线的交点就是最后的误差函数停止的值
    """
    return np.mean((np.dot(feature, theta) - label) ** 2) * 0.5


def gradient_descent(path, number, theta, feature, label, learn_rate):
    """
    训练指定次数，返回损失函数和每一步位置
    number：训练次数
    theta：初始theta
    """
    cost_list = list()
    for _ in range(number):
        cost = _get_cost(theta, feature, label)
        theta = _get_theta(theta, feature, label, learn_rate)
        cost_list.append(cost)
    with open(path, 'w', encoding='utf-8') as f:
        for i in theta:
            for j in i:
                f.write(str(j) + '\n')
    return cost_list


def test_model(theta, feature, label):
    """
    使用R方误差来测试模型的优劣
    越接近1越优
    """
    r = 1 - _get_cost(theta, feature, label) / np.var(label)
    return r
