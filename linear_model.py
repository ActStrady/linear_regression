import numpy as np
"""
一个多元线性回归的训练模型
"""
def get_grad(theta, x, y):
    """
    梯度计算,实际上就是求了偏导
    theta 
    x 给定的值
    y 给定的值
    return -grad
    """
    grad = 2 * (theta * x - y) * x
    return grad

def get_theta(theta, x, y, learn_ratio):
    """
    计算下一theta
    theta 前一theta
    learn_ratio 学习率，其实就是步长
    """
    return theta + get_grad(theta, x, y) * learn_ratio

def get_cost(theta, x, y):
    """
    误差函数，衡量拟合函数
    """
    return np.sum((np.dot(x, theta) - y) ** 2)

if __name__ == "__main__":
    theta = np.array([1, 1])
    x = np.array([[1, 2], [1,2]])
    y = np.array([0, 0])
    learn_ratio = 0.1
    for _ in range(20):
        print(get_cost(theta, x, y))
        theta = get_theta(theta, x, y, learn_ratio)


