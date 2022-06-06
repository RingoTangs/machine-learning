#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def create_data_set() -> tuple:
    # 四组二维特征
    group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])
    # 四组特征的标签
    labels = ['爱情片', '爱情片', '动作片', '动作片']
    return group, labels


def classify(in_x, data_set: np.ndarray, labels, k):
    """
    函数说明: KNN 算法, 分类器
    :param in_x: 用于分类的数据（测试集）
    :param data_set: 用于训练的数据（训练集）
    :param labels: 分类标签
    :param k: KNN 算法参数, 选择距离最小的 k 个点
    :return: sortedClassCount[0][0] - 分类结果
    """
    # numpy.array 数组的行数
    data_set_size = data_set.shape[0]
    # in_x 横向复制一次, 纵向重复 data_set_size 次。然后进行矩阵减法
    diff_mat = np.tile(in_x, (data_set_size, 1)) - data_set
    # 数组中的每个数都取平方
    sq_diff_mat = diff_mat ** 2
    # 每一行的元素相加, 得到一维数组
    sq_distances = sq_diff_mat.sum(axis=1)
    # 开方计算距离
    distances = sq_distances ** 0.5
    # 返回 distances 中元素从小到大排列后的索引
    sorted_distance_indices = distances.argsort()
    # 定义一个记录类别次数的字典
    class_count = {}

    # 取出前 k 个元素的类别
    for i in range(k):
        vote_i_label = labels[sorted_distance_indices[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    # 对字典进行排序（降序）=> 返回格式：[('动作片', 2), ('爱情片', 1)]
    sorted_class_count = sorted(class_count.items(), key=lambda a: a[1], reverse=True)
    return sorted_class_count[0][0]


if __name__ == '__main__':
    test = [101, 20]
    group, labels = create_data_set()
    k = 3
    result = classify(test, group, labels, k)
    print(result)
    plt.scatter(group[:, 0:1], group[:, 1:2])
    plt.scatter(x=test[0], y=test[1], c='red')
    plt.title(u'散点图', color='red')
    plt.show()
