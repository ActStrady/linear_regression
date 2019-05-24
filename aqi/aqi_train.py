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


# TODO 模型训练和验证以及预测的后续整合
# 训练模型
def theta_aqi(path, feature, label):
    # 初始theta
    theta = np.zeros((6, 1))
    # 训练次数
    number = 300
    # 学习率
    learn_rate = 0.00001
    # 训练得出损失函数值
    cost_list = linear_model.gradient_descent(path, number, theta, feature, label, learn_rate)
    return cost_list


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
    with open('theta.txt', 'r') as f:
        theta = np.array([float(line) for line in f.readlines()]).reshape(6, 1)
    return np.dot(feature, theta)


if __name__ == '__main__':
    aqi_data = _deal_aqi_data()
    path = 'theta.txt'
    # 训练数据
    train_feature, train_label = aqi_data[0]
    theta_aqi(path, train_feature, train_label)
    # 预测
    # aqi = linear_model.get_aqi_value([20, 46, 8, 37, 0.51, 75], theta)
    # print(aqi)
