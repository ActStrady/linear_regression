"""
处理数据：标准化，读取数据，拆分数据
"""

import numpy as np
import pandas as pd


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


def standardized(data):
    """
    标准化标签,
    """
    # 最大最小值标准化
    # standardized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    # 方差标准化
    # standardized_data = (data - np.mean(data)) / np.std(data)
    # lg 标准化
    standardized_data = np.log(data)
    return standardized_data


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
    train_feature = features[:train_rows]
    train_label = label[:train_rows]
    # 验证数据
    verification_feature = features[train_rows:train_rows + verification_rows]
    verification_label = label[train_rows:train_rows + verification_rows]
    # 测试数据
    test_feature = features[verification_rows:]
    test_label = label[verification_rows:]
    return (train_feature, train_label), (verification_feature, verification_label), (test_feature, test_label)
