"""
一个多元线性回归的训练模型
利用了梯度下降算法，具体公式参考梯度下降算法
grad: 表示梯度
tenta: 表示下降后每一次的位置， 一般初始位置设置为0
cost: 表示损失函数，就是与真实值的差距，一般设置为一个二元函数
featurs: 特征值, 就是要参与训练的值，简单来说就是利用这些值来拟合出曲线，对应数学公式的x
lable: 标签， 就是要预测的那条数据，就是实际值，对应数学公式中的y
learn_rate: 学习率，就是步长，每次下降的频度
"""

import numpy as np


def _get_grad(theta, featurs, lable):
    """
    梯度计算,利用了偏导
    由于使用了矩阵，就需要对featurs进行转置，因为数学公式中引入矩阵后推导后的结果就是这样
    """
    grad = np.dot(np.transpose(featurs), (np.dot(featurs, theta) - lable))
    return grad


def _get_theta(theta, featurs, lable, learn_rate):
    """
    计算下一theta
    按照公式这里应该除len(featurs)，但考虑到计算机计算除法比较慢，这里只是一种思想，所以不加也可以
    """
    return theta - _get_grad(theta, featurs, lable,) * learn_rate


def _get_cost(theta, featurs, lable):
    """
    误差函数，衡量拟合函数的效果
    误差会逐步逼近0
    一般训练数据与验证数据算出的曲线的交点就是最后的误差函数停止的值
    """
    return np.mean((np.dot(featurs, theta) - lable) ** 2) * 0.5


def gradient_descent(number, theta, featurs, lable, learn_rate):
    cost_list = list()
    theta_list = list()
    for _ in number:
        cost = _get_cost(theta, featurs, lable)
        theta = _get_theta(theta, featurs, lable, learn_rate)
        cost_list.append(cost)
        theta_list.append(theta)
    return cost_list, theta_list




if __name__ == "__main__":
    theta = np.array([1, 1])
    x = np.array([[1, 2], [1, 2]])
    y = np.array([0, 0])
    learn_ratio = 0.1
    for _ in range(20):
        print(get_cost(theta, x, y))
        theta = get_theta(theta, x, y, learn_ratio)
