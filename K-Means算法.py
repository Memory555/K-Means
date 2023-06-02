import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 #有中文出现的情况，需要u'内容'
import pandas as pd

# 欧氏距离计算
def distEclud(x, y):
    return np.sqrt(np.sum((x - y) ** 2))  # 计算欧氏距离


# 为给定数据集构建一个包含K个随机质心centroids的集合
def randCent(dataSet, k):
    np.random.seed()
    m, n = dataSet.shape  # m=150,n=4
    centroids = np.zeros((k, n))  # 4*4
    for i in range(k):  # 执行四次
        index = int(np.random.uniform(0, m))  # 产生0到150的随机数（在数据集中随机挑一个向量做为质心的初值）
        centroids[i, :] = dataSet[index, :]  # 把对应行的四个维度传给质心的集合
    return centroids


# k均值聚类算法
def KMeans(dataSet, k):
    m = np.shape(dataSet)[0]  # 行数150
    # 第一列存每个样本属于哪一簇(四个簇)
    # 第二列存每个样本的到簇的中心点的误差
    clusterAssment = np.mat(np.zeros((m, 2)))  # .mat()创建150*2的矩阵
    clusterChange = True

    # 1.初始化质心centroids
    centroids = randCent(dataSet, k)  # 4*4
    while clusterChange:
        # 样本所属簇不再更新时停止迭代
        clusterChange = False

        # 遍历所有的样本（行数150）
        for i in range(m):
            minDist = 100000.0
            minIndex = -1

            # 遍历所有的质心
            # 2.找出最近的质心
            for j in range(k):
                # 计算该样本到4个质心的欧式距离，找到距离最近的那个质心minIndex
                distance = distEclud(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # 3.更新该行样本所属的簇
            if clusterAssment[i, 0] != minIndex:
                clusterChange = True
                clusterAssment[i, :] = minIndex, minDist ** 2
        # 4.更新质心
        # np.nonzero(x)返回值不为零的元素的下标，它的返回值是一个长度为x.ndim(x的轴数)的元组
        # 元组的每个元素都是一个整数数组，其值为非零元素的下标在对应轴上的值。
        # 矩阵名.A 代表将 矩阵转化为array数组类型

        # 这里取矩阵clusterAssment所有行的第一列，转为一个array数组，与j（簇类标签值）比较，返回true or false
        # 通过np.nonzero产生一个array，其中是对应簇类所有的点的下标值（x个）
        # 再用这些下标值求出dataSet数据集中的对应行，保存为pointsInCluster（x*4）
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]  # 获取对应簇类所有的点（x*4）
            centroids[j, :] = np.mean(pointsInCluster, axis=0)  # 求均值，产生新的质心
    return centroids, clusterAssment


def draw(data, center, assment):
    length = len(center)
    data1 = data[np.nonzero(assment[:, 0].A == 0)[0]]
    data2 = data[np.nonzero(assment[:, 0].A == 1)[0]]
    data3 = data[np.nonzero(assment[:, 0].A == 2)[0]]
    # 选取前两个维度绘制原始数据的散点图
    plt.scatter(data1[:, 0], data1[:, 1], c="red", marker='x', label='label0')
    plt.scatter(data2[:, 0], data2[:, 1], c="green", marker='*', label='label1')
    plt.scatter(data3[:, 0], data3[:, 1], c="blue", marker='+', label='label2')
    # # 绘制簇的质心点
    for i in range(length):
        plt.scatter(center[i, 0], center[i, 1], c="black", marker='o')
    plt.legend()
    plt.title("花萼长度和宽度关系聚类划分结果")
    plt.show()

    # 选取后两个维度绘制原始数据的散点图
    plt.scatter(data1[:, 2], data1[:, 3], c="red", marker='x', label='label0')
    plt.scatter(data2[:, 2], data2[:, 3], c="green", marker='*', label='label1')
    plt.scatter(data3[:, 2], data3[:, 3], c="blue", marker='+', label='label2')
    # 绘制簇的质心点
    for i in range(length):
        plt.scatter(center[i, 2], center[i, 3], c="black", marker='o')
    plt.legend()
    plt.title("花瓣长度和宽度关系聚类划分结果")
    plt.show()

if __name__ == '__main__':
    dataset = pd.read_csv('./iris.csv')
    dataset.drop(dataset.columns[0], axis=1, inplace=True)  # 删除第1列
    # 取150个样本，取前两列特征，花萼长度和宽度
    X = np.array(dataset.iloc[:, 0:4])  # 能看特征数据的具体信息
    y = np.array(dataset.iloc[:, 4])  # 能看每行数据的标签的值
    label = []
    for i in y:
        if i == 'setosa':
            label.append(0)
        elif i == 'versicolor':
            label.append(1)
        elif i == 'virginica':
            label.append(2)
    dataSet = X
    k = 3
    centroids, clusterAssment = KMeans(dataSet, k)
    pre_label = clusterAssment[:,0]
    count = 0
    for j in range(len(label)):
        if label[j] == pre_label[j]:
            count = count + 1
    print("划分精确度",count/(len(pre_label)))
    draw(dataSet, centroids, clusterAssment)
