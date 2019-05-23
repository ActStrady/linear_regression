"""
处理数据：标准化，读取数据，拆分数据
"""

import pandas as pd
import numpy as np

label_name = 'AQI'
features_name = ['PM2.5', 'PM10', 'CO', 'No2', 'So2', 'O3']


def read_data(path, features_name, label_name):
    """
    读取数据
    path: 路径
    features_name：特征名
    label_name： 标签名
    """
    aqi_data = pd.read_csv(path, encoding='utf-8')
    # 标签, 转换为n行1列的数据
    label = aqi_data[label_name].values.reshape(-1, 1)
    # 特征
    features = aqi_data[features_name].values
    return features, label


def standardized(features):
    '''
    标准化标签
    '''
    # 标准化，最大最小值标准化
    features = (features - np.min(features)) / (np.max(features) - np.min(features))
    # 标准化, 方差标准化
    # features = (features - np.mean(features)) / np.std(features)
    return features


def break_up(features, label, train_proportion, verification_proportion):
    """
    拆分数据为三块，训练，验证，测试
    train_proportion: 训练数据占比
    verification_proportion： 验证数据占比
    """
    # 数据占比
    rows = len(features)
    train_rows = int(rows * train_proportion)
    verification_rows = int(rows * verification_proportion)
    # 训练数据
    train_featurs = features[:train_rows]
    train_lable = label[:train_rows]
    # 验证数据
    verification_featurs = features[train_rows:train_rows + verification_rows]
    verification_lable = label[train_rows:train_rows + verification_rows]
    # 测试数据
    test_featurs = features[verification_lable:]
    test_lable = label[verification_lable:]
    return (train_featurs, train_lable), (verification_featurs, verification_lable), (test_featurs, test_lable)
