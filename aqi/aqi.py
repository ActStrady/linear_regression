"""
利用回归
"""
import numpy as np

from show import show_cost
from read_data import read_aqi


def get_grad(theta, x, y):
    grad = np.dot(np.transpose(x), (np.dot(x, theta) - y))
    return grad


def get_theta(theta, x, y, learn_ratio):
    return theta - get_grad(theta, x, y) * learn_ratio / len(x)


def get_cost(theta, x, y):
    return np.mean((np.dot(x, theta) - y) ** 2) * 0.5


if __name__ == "__main__":
    aqi_data = read_aqi('aqi2.csv')
    train_featurs, train_lable = aqi_data[0]
    verification_featurs, verification_lable = aqi_data[1]
    test_featurs, test_lable = aqi_data[2]
    
    # 初始theta
    theta = np.zeros((6, 1))
    learn_ratio = 0.0001
    
    train_cost_list = list()
    verification_list = list()
    test_list = list()
    for _ in range(200):
        # 
        theta = np.zeros((6, 1))
        train_cost = get_cost(theta, train_featurs, train_lable)
        train_cost_list.append(train_cost)
        theta = get_theta(theta, train_featurs, train_lable, learn_ratio)

        theta = np.zeros((6, 1))
        verification_cost = get_cost(theta, verification_featurs, verification_lable)
        verification_list.append(verification_cost)
        theta = get_theta(theta, verification_featurs, verification_lable, learn_ratio)

        theta = np.zeros((6, 1))
        test_cost = get_cost(theta, test_featurs, test_lable)
        test_list.append(test_cost)
        theta = get_theta(theta, test_featurs, test_lable, learn_ratio)
    show_cost(train_cost_list, verification_list, test_list)
