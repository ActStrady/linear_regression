import numpy as np
from util import show
from util import deal_data
from util import linear_model


# 处理数据
def _deal_aqi_data():
    # 标签名
    label_name = 'AQI'
    # 特征值名
    features_name = ['PM2.5', 'PM10', 'CO', 'No2', 'So2', 'O3']
    # 读取数据
    features, label = deal_data.read_data('aqi.csv', features_name, label_name)
    # 标准化数据
    features = deal_data.standardized(features)
    # 拆分数据并返回
    return deal_data.break_up(features, label, 0.6, 0.3)


# 训练模型
def theta_aqi():
    path = 'theta.txt'
    aqi_data = _deal_aqi_data()
    # 初始theta
    theta = np.zeros((6, 1))
    # 学习率
    learn_rate = 0.00001
    return linear_model.gradient_descent(path, theta, aqi_data, learn_rate)


def get_aqi_value(feature):
    """
    预测aqi值
    :param feature: 输入的特征值
    :return: 预测的aqi值
    """
    # 转化为矩阵
    feature = np.array(feature)
    # 标准化处理
    feature = deal_data.standardized(feature)
    # 从文件中读取训练后的theta
    with open('theta.txt', 'r') as f:
        theta = np.array([float(line) for line in f.readlines()]).reshape(6, 1)
    return np.dot(feature, theta)


if __name__ == '__main__':
    # 训练数据
    cost_lists = theta_aqi()
    show.show_cost(cost_lists)
    # 预测
    aqi = get_aqi_value([48, 94, 14, 37, 0.75, 133])
    print(aqi)
