# -*- coding:utf-8 -*-
"""
通过seaborn输出各种图形
"""
import seaborn as sns
import matplotlib.pyplot as plt


def show_cost(data_lists):
    """
    输出梯度下降过程中损失函数值的变化情况
    """
    sns.set_style("whitegrid")
    for data_list in data_lists:
        plt.plot(data_list)
    plt.show()
