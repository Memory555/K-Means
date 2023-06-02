import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 #有中文出现的情况，需要u'内容'

def show():
    # 两两关系
    dataset = pd.read_csv('./iris.csv')
    dataset.drop(dataset.columns[0], axis=1, inplace=True)  # 删除第1列
    ##取150个样本，取前两列特征，花萼长度和宽度
    x_1 = np.array(dataset.iloc[:, 0:2])  # 能看特征数据的具体信息
    y = np.array(dataset.iloc[:, 4])  # 能看每行数据的标签的值
    ##分别取前两类样本，0和1
    samples_0 = x_1[y == 'setosa', :]  # 把y=0,即Iris-setosa的样本取出来
    samples_1 = x_1[y == 'versicolor', :]  # 把y=1，即Iris-versicolo的样本取出来
    samples_2 = x_1[y == 'virginica', :]  # 把y=2，即Iris-virginica的样本取出来

    # 散点图可视化
    plt.subplot(1, 2, 1)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 #有中文出现的情况，需要u'内容'
    plt.scatter(samples_0[:, 0], samples_0[:, 1], marker='o', color='r', label='Iris Setosa')
    plt.scatter(samples_1[:, 0], samples_1[:, 1], marker='x', color='b', label='Iris Versicolor')
    plt.scatter(samples_2[:, 0], samples_2[:, 1], marker='*', color='y', label='Iris Virginica]')
    plt.xlabel('花萼长度', fontsize=14)
    plt.ylabel('花萼宽度', fontsize=14)
    plt.legend()

    x_2 = np.array(dataset.iloc[:, 2:4])  # 能看特征数据的具体信息
    ##分别取前两类样本，0和1
    samples_0 = x_2[y == 'setosa', :]  # 把y=0,即Iris-setosa的样本取出来
    samples_1 = x_2[y == 'versicolor', :]  # 把y=1，即Iris-versicolo的样本取出来
    samples_2 = x_2[y == 'virginica', :]  # 把y=2，即Iris-virginica的样本取出来
    # 散点图可视化
    plt.subplot(1, 2, 2)
    plt.scatter(samples_0[:, 0], samples_0[:, 1], marker='o', color='r', label='Iris Setosa')
    plt.scatter(samples_1[:, 0], samples_1[:, 1], marker='x', color='b', label='Iris Versicolor')
    plt.scatter(samples_2[:, 0], samples_2[:, 1], marker='*', color='y', label='Iris Virginica]')
    plt.xlabel('花瓣长度', fontsize=14)
    plt.ylabel('花瓣宽度', fontsize=14)
    plt.legend()
    plt.show()

    # print(dataset)
    sns.despine()  # 去坐标轴
    sns.set(style='white', color_codes=True)  # 设置样式
    plt.subplot(2, 2, 1)
    # plt.title('sepal width')
    sns.stripplot(x=dataset['Species'], y=dataset['Sepal.Width'])

    plt.subplot(2, 2, 2)
    # plt.title('sepal length')
    sns.stripplot(x=dataset['Species'], y=dataset['Sepal.Length'])

    plt.subplot(2, 2, 3)
    # plt.title('petal width')
    sns.stripplot(x=dataset['Species'], y=dataset['Petal.Width'])

    plt.subplot(2, 2, 4)
    # plt.title('petal length')
    sns.stripplot(x=dataset['Species'], y=dataset['Petal.Length'])
    plt.show()

    return dataset


if __name__ == '__main__':
    show()
