#!/usr/bin/python
# -*- coding: UTF-8 -*-
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

"""
k-近临算法的一般流程：
1. 收集数据：可以使用爬虫进行数据的收集，也可以使用第三方提供的免费或收费的数据。一般来讲，数据放在txt文本文件中，按照一定的格式进行存储，便于解析及处理。
2. 准备数据：使用Python解析、预处理数据。
3. 分析数据：可以使用很多方法对数据进行分析，例如使用Matplotlib将数据可视化。
4. 测试算法：计算错误率。
5. 使用算法：错误率在可接受范围内，就可以运行k-近邻算法进行分类。

背景信息:
海伦女士一直使用在线约会网站寻找适合自己的约会对象。尽管约会网站会推荐不同的任选，但她并不是喜欢每一个人。经过一番总结，她发现自己交往过的人可以进行如下分类：
1. 不喜欢的人（didntLike）
2. 魅力一般的人（smallDoses）
3. 极具魅力的人（largeDoses）

海伦收集的样本数据主要包含以下3种特征：
1. 每年获得的飞行常客里程数
2. 玩视频游戏所消耗时间百分比
3. 每周消费的冰淇淋公升数
"""


def file2matrix(filename: str) -> tuple:
    """
    函数说明: 打开并解析文件, 对数据进行分类: 1代表不喜欢, 2代表魅力一般, 3代表极具魅力。
    :param filename - 文件名
    :return: return_matrix - 特征矩阵
             class_label_vector - 分类 Label 向量
    """
    # 打开文件
    fr = open(filename, mode='r')
    # 读取文件所有内容
    lines = fr.readlines()
    # 得到文件行数
    number_of_lines = len(lines)
    # 返回的 Numpy 矩阵. number_of_lines 行, 3 列
    dating_matrix = np.zeros((number_of_lines, 3), dtype=float)
    # 返回的分类标签向量
    dating_labels = []
    # 行的索引值
    index = 0
    for line in lines:
        # str.strip(rm), 不写 rm, 默认删除空白字符（包括'\n', '\r', '\t', ' '）
        line = line.strip()
        # 字符串分割
        list_from_line = line.split('\t')
        # 将前3列提取出来, 存放到 return_matrix 的 Numpy 矩阵中
        dating_matrix[index, :] = list_from_line[0:3]
        # 根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if list_from_line[-1] == 'didntLike':
            dating_labels.append(1)
        elif list_from_line[-1] == 'smallDoses':
            dating_labels.append(2)
        elif list_from_line[-1] == 'largeDoses':
            dating_labels.append(3)
        index += 1
    return dating_matrix, dating_labels


def show_data(dating_matrix: np.ndarray, dating_labels: list) -> None:
    """
    函数说明: 数据可视化
    :param dating_matrix - 特征矩阵
    :param dating_labels - 分类 Label
    :return: None
    """
    # 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    # 当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
    fig, ((axs0, axs1), (axs2, axs3)) = plt.subplots(nrows=2, ncols=2, figsize=(13, 8))
    # Label 所对应的颜色
    labels_colors = []
    for i in dating_labels:
        if i == 1:
            labels_colors.append('black')
        elif i == 2:
            labels_colors.append('orange')
        elif i == 3:
            labels_colors.append('red')
    print(dating_labels)
    print(labels_colors)
    # 画出散点图,以 dating_matrix 矩阵的第一列(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs0.scatter(x=dating_matrix[:, 0], y=dating_matrix[:, 1], c=labels_colors, s=15, alpha=.5)
    # 设置标题, x轴label, y轴label
    axs0_title_text = axs0.set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比')
    axs0_x_label_text = axs0.set_xlabel(u'每年获得的飞行常客里程数')
    axs0_y_label_text = axs0.set_ylabel(u'玩视频游戏所消耗时间占比')
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_x_label_text, size=7, weight='bold', color='black')
    plt.setp(axs0_y_label_text, size=7, weight='bold', color='black')

    # 画出散点图,以 dating_matrix 矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs1.scatter(x=dating_matrix[:, 0], y=dating_matrix[:, 2], c=labels_colors, s=15, alpha=.5)
    axs1_title_text = axs1.set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数')
    axs1_x_label_text = axs1.set_xlabel(u'每年获得的飞行常客里程数')
    axs1_y_label_text = axs1.set_ylabel(u'每周消费的冰激淋公升数')
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_x_label_text, size=7, weight='bold', color='black')
    plt.setp(axs1_y_label_text, size=7, weight='bold', color='black')

    # 画出散点图,以 dating_matrix 矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs2.scatter(x=dating_matrix[:, 1], y=dating_matrix[:, 2], c=labels_colors, s=15, alpha=.5)
    axs2_title_text = axs2.set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数')
    axs2_x_label_text = axs2.set_xlabel(u'玩视频游戏所消耗时间占比')
    axs2_y_label_text = axs2.set_ylabel(u'每周消费的冰激淋公升数')
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_x_label_text, size=7, weight='bold', color='black')
    plt.setp(axs2_y_label_text, size=7, weight='bold', color='black')

    # 设置图例
    didnt_like = mlines.Line2D([], [], color='black', marker='.', markersize=6, label='didntLike')
    small_doses = mlines.Line2D([], [], color='orange', marker='.', markersize=6, label='smallDoses')
    large_doses = mlines.Line2D([], [], color='red', marker='.', markersize=6, label='largeDoses')
    axs0.legend(handles=[didnt_like, small_doses, large_doses])
    axs1.legend(handles=[didnt_like, small_doses, large_doses])
    axs2.legend(handles=[didnt_like, small_doses, large_doses])

    # 显示图片
    plt.show()


if __name__ == '__main__':
    filename = './sample2-data-set.txt'
    dating_matrix, dating_labels = file2matrix(filename)
    show_data(dating_matrix, dating_labels)
