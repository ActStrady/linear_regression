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
    实际上是一个均方误差（RMSE）
    误差会逐步逼近0
    一般训练数据与验证数据算出的曲线的交点就是最后的误差函数停止的值
    """
    return np.mean((np.dot(feature, theta) - label) ** 2) * 0.5


def gradient_descent(path, theta, data, learn_rate):
    """
    梯度下降算法来训练模型
    何时停止？一般当验证集和训练集数据训练的损失函数值相交
    现在由于学习率无法调整，故先10000次
    :theta 初始theta
    :return 损失函数值列表
    """
    # 损失函数值结果集
    train_cost_list = list()
    verification_cost_list = list()
    test_cost_list = list()
    # 初始theta
    train_theta = theta
    verification_theta = theta
    test_theta = theta
    # TODO 验证集和训练集数据训练的损失函数值相交处理
    for _ in range(10000):
        # 训练集
        train_feature, train_label = data[0]
        # 训练损失函数值
        train_cost = _get_cost(train_theta, train_feature, train_label)
        train_theta = _get_theta(train_theta, train_feature, train_label, learn_rate)
        # 添加损失函数值结果集
        train_cost_list.append(train_cost)

        # 验证集
        verification_feature, verification_label = data[1]
        verification_cost = _get_cost(verification_theta, verification_feature, verification_label)
        verification_theta = _get_theta(verification_theta, verification_feature, verification_label, learn_rate)
        verification_cost_list.append(verification_cost)

        # 测试集
        test_feature, test_label = data[2]
        test_cost = test_model(test_theta, test_feature, test_label)
        print(test_cost)
        test_theta = _get_theta(test_theta, test_feature, test_label, learn_rate)
        test_cost_list.append(test_cost)

    # 将训练最后的theta保存到文件
    with open(path, 'w', encoding='utf-8') as f:
        for i in train_theta:
            for j in i:
                f.write(str(j) + '\n')
    return train_cost_list, verification_cost_list, test_cost_list


def test_model(theta, feature, label):
    """
    使用R方误差来测试模型的优劣
    越接近1越优
    """
    return 1 - _get_cost(theta, feature, label) / np.var(label)