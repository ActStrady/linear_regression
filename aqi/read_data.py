# -*- coding:utf-8 -*-
"""
读取数据
"""

import pandas as pd
import numpy as np

def read_aqi(path):
    # 读取数据
    aqi_data = pd.read_csv(path, encoding='utf-8')
    # 标签, 转换为n行1列的数据
    label = aqi_data["AQI"].values.reshape(-1, 1)
    cols = ['PM2.5', 'PM10', 'CO', 'No2', 'So2', 'O3']
    # 特征
    features = aqi_data[cols].values

    # 标准化，最大最小值标准化
    features = (features - np.min(features)) / (np.max(features) - np.min(features))
    # 标准化, 方差标准化
    # features = (features - np.mean(features)) / np.std(features)

    # 拆分数据为三块，训练， 验证，测试
    rows = len(features)
    # 训练数据
    train_rows = int(rows * 0.6)
    # 验证数据
    verification_rows = int(rows * 0.3)
    train_featurs, train_lable = features[:train_rows], label[:train_rows]
    verification_featurs, verification_lable = features[train_rows:train_rows + verification_rows], label[train_rows:train_rows + verification_rows]
    test_featurs, test_lable = features[verification_lable:], label[verification_lable:]
    return (train_featurs, train_lable), (verification_featurs, verification_lable), (test_featurs, test_lable)
