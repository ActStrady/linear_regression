import util.show as show
import util.deal_data as delta
import util.linear_model as lin
import numpy as np


# 处理数据
label_name = 'AQI'
features_name = ['PM2.5', 'PM10', 'CO', 'No2', 'So2', 'O3']
features, label = delta.read_data('aqi.csv', features_name, label_name)
features = delta.standardized(features)
print(features)
aqi_data = delta.break_up(features, label, 0.6, 0.3)
print(aqi_data)
theta = np.zeros((6, 1))