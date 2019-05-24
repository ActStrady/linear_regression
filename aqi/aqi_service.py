#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# @Time : 2019/5/24 14:23
# @Author : ActStrady@tom.com
# @FileName : aqi_service.py
# @GitHub : https://github.com/ActStrady/linear_regression
from flask import Flask, request
from flask_cors import CORS
import aqi.aqi_train as aqt
import json

APP = Flask(__name__)
CORS(APP)


@APP.route('/aqi', methods=['GET', 'POST'])
def aqi():
    json_data = request.get_data()
    json_data = json.loads(json_data.decode('utf-8'))
    pm25 = json_data.get('pm25')
    pm10 = json_data.get('pm10')
    co = json_data.get('co')
    no2 = json_data.get('no2')
    so2 = json_data.get('so2')
    o3 = json_data.get('o3')
    input_data = [pm25, pm10, co, no2, so2, o3]
    aqi_value = aqt.get_aqi_value(input_data)
    return json.dumps({'result': aqi_value[0]})


if __name__ == '__main__':
    APP.run()
