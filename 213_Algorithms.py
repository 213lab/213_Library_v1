"""
开发版本：v1
开发作者：吴舜禹
开发时间：2021.3.3
库功能： 1.聚类算法（Kmeans、DBSCAM）
        2.爬虫算法
        3.lenet5卷积网络（简易图像识别）
        4.LSTM简单网络结构
        5.预测方法（细粒度预测、加权马尔科夫链）
        6.Transformer（整理中）
"""

import numpy as np
import pandas as pd
import random
import torch.nn as nn
import torch.nn.functional as  F
import re
import time
import os
import requests
from bs4 import BeautifulSoup
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

print('test')
class cluster:
    def k_means(self, data, K, tol, N):
        n = np.shape(data)[0]
        centerId = random.sample(range(0, n), K)
        centerPoints = data[centerId]
        dist = cdist(data, centerPoints, metric='euclidean')
        labels = np.argmin(dist, axis=1).squeeze()
        oldVar = -0.0001
        newVar = np.sum(np.sqrt(np.sum(np.power(data - centerPoints[labels], 2), axis=1)))
        count = 0
        while count < N and abs(newVar - oldVar) > tol:
            oldVar = newVar
            for i in range(K):
                centerPoints[i] = np.mean(data[np.where(labels == i)], 0)
            dist = cdist(data, centerPoints, metric='euclidean')
            labels = np.argmin(dist, axis=1).squeeze()
            newVar = np.sum(np.sqrt(np.sum(np.power(data - centerPoints[labels], 2), axis=1)))
            count += 1
        return labels, centerPoints

    def DBSCAN(self, data, eps, minPts):
        disMat = squareform(pdist(data, metric='euclidean'))
        n, m = data.shape
        core_points_index = np.where(np.sum(np.where(disMat <= eps, 1, 0), axis=1) >= minPts)[0]
        labels = np.full((n,), -1)
        clusterId = 0
        for pointId in core_points_index:
            if labels[pointId] == -1:
                labels[pointId] = clusterId
                neighbour = np.where((disMat[:, pointId] <= eps) & (labels == -1))[0]
                seeds = set(neighbour)
                while len(seeds) > 0:
                    newPoint = seeds.pop()
                    labels[newPoint] = clusterId
                    queryResults = np.where(disMat[:, newPoint] <= eps)[0]
                    if len(queryResults) >= minPts:
                        for resultPoint in queryResults:
                            if labels[resultPoint] == -1:
                                seeds.add(resultPoint)
                clusterId = clusterId + 1
        return labels

class crawler:
    def get_html(self, url):
        try:
            # 添加User-Agent，放在headers中，伪装成浏览器
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.162 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            # print(response.status_code)
            if response.status_code == 200:
                response.encoding = 'utf-8'
                return response.text
            return None
        except requests.exceptions:
            print("ERROR!!!Check your code")
            return None

    def get_url_list(self, detailed_html, html):
        # 将get_page函数中获取页面中的会议论文的URL信息解析出来
        url_list = []
        pattern = re.compile("this.id,'(.*?)'", re.S)
        ids = pattern.findall(html)
        for id in ids:
            url_list.append(detailed_html + id)
        return url_list

    def get_information(self, url, single_index, area_index):
        details = self.get_html(url)
        # 使用beautifulSoup进行解析
        soup = BeautifulSoup(details, 'lxml')
        # 题目
        single_text = soup.select(single_index)[0].text
        # 摘要
        area_text = soup.select(area_index)[0].textarea
        if area_text:
            area_text = area_text.text.strip()
        else:
            area_text = ''

        information = [single_text, area_text]
        return information

    def go_cawler(self, search_url, detailed_html, single_index, area_index):
        html = self.get_html(search_url)
        url_list = self.get_url_list(detailed_html, html)
        for url in url_list:
            information = self.get_information(url, single_index, area_index)
            time.sleep(random.uniform(1, 2))
        return information

class leNet5(nn.Module):
    # 定义Net的初始化函数，这个函数定义了该神经网络的基本结构
    def __init__(self, num_class):
        super(leNet5, self).__init__()
        # 复制并使用Net的父类的初始化方法，即先运行nn.Module的初始化函数
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 定义conv1函数的是图像卷积函数：输入为图像（1个频道，即灰度图）,输出为 6张特征图, 卷积核为5x5正方形
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 定义conv2函数的是图像卷积函数：输入为6张特征图,输出为16张特征图, 卷积核为5x5正方形
        self.fc1 = nn.Linear(2704, 120)  # 为什么时2704至今没想明白
        # 定义fc1（fullconnect）全连接函数1为线性函数：y = Wx + b，并将16*5*5个节点连接到120个节点上。
        self.fc2 = nn.Linear(120, 84)
        # 定义fc2（fullconnect）全连接函数2为线性函数：y = Wx + b，并将120个节点连接到84个节点上。
        self.fc3 = nn.Linear(84, num_class)
        # 定义fc3（fullconnect）全连接函数3为线性函数：y = Wx + b，并将84个节点连接到10个节点上。

    # 定义该神经网络的向前传播函数，该函数必须定义，一旦定义成功，向后传播函数也会自动生成（autograd）
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 输入x经过卷积conv1之后，经过激活函数ReLU，使用2x2的窗口进行最大池化Max pooling，然后更新到x。
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # 输入x经过卷积conv2之后，经过激活函数ReLU，使用2x2的窗口进行最大池化Max pooling，然后更新到x。
        x = x.view(-1, self.num_flat_features(x))
        # view函数将张量x变形成一维的向量形式，总特征数并不改变，为接下来的全连接作准备。
        x = F.relu(self.fc1(x))
        # 输入x经过全连接1，再经过ReLU激活函数，然后更新x
        x = F.relu(self.fc2(x))
        # 输入x经过全连接2，再经过ReLU激活函数，然后更新x
        x = self.fc3(x)
        # 输入x经过全连接3，然后更新x
        return x

    # 使用num_flat_features函数计算张量x的总特征量（把每个数字都看出是一个特征，即特征总量），比如x是4*2*2的张量，那么它的特征总量就是16。
    def num_flat_features(self, x):
        size = x.size()[1:]
        # 这里为什么要使用[1:],是因为pytorch只接受批输入，也就是说一次性输入好几张图片，那么输入数据张量的维度自然上升到了4维。【1:】让我们把注意力放在后3维上面
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class forecasting:
    def fine_grained_forecasting(self, data):
        date = data['TIME']
        flow = data['0']
        cum_flow = pd.DataFrame(np.zeros(int(len(date) / 1440)))  # 行数表示每天的累计流量
        count = 0
        flag = np.zeros(int(len(date) / 1440), dtype=int)
        date = pd.to_datetime(date)
        date = date.dt.strftime('%Y:%m:%d')  # 格式转换，消去小数。
        print('时间为：\n{}'.format(date))
        print('瞬时值为：\n{}'.format(flow))

        # 累积流量计算
        for i in range(len(date)):
            # print(date[i])
            # print(i)
            if i >= 1 and date[i] != date[i - 1]:  # 找到新一天的索引
                # print(flag[count])
                cum_flow.iloc[count] = sum(flow[flag[count]: i])  # 补上最后一天累计流量
                count += 1
                flag[count] = i  # 每更新一天，flag也随之更新一天
            # print(count)
        aim_day = 115
        print('累积流量为：\n{}'.format(cum_flow))

        flow_num = 1440
        forecasting_flow = pd.DataFrame(np.zeros([7, flow_num]))

        for index in range(7):
            delta = pd.DataFrame(np.zeros(aim_day + index - 1))
            for t in range(aim_day + index - 1):
                delta.iloc[t] = abs(cum_flow.iloc[t] - cum_flow.iloc[aim_day + index])
            print('当前为周{}，差值为{}'.format(index + 1, delta))
            closest = delta.sort_values(by=[0], ascending=True)
            closest_weekends = 0
            count = 0
            for j in range(3):
                if closest.index[j] % 7 == 4 or closest.index[j] % 7 == 5:
                    closest_weekends += 1
            print('最接近的几天为：\n{}'.format(closest))
            for i in range(len(date)):
                # print(i)
                if i >= 1 and date[i] != date[i - 1]:  # 到新的一天了
                    # print(count, closest.index[0], closest.index[1], closest.index[2])
                    if count == closest.index[0] or count == closest.index[1] or count == closest.index[2]:
                        if (aim_day + index) % 7 == 4 or (aim_day + index) % 7 == 5:
                            if closest_weekends == 0:
                                for j in range(flow_num):  # 若是则挨个取平均
                                    forecasting_flow.ix[index, j] = forecasting_flow.ix[index, j] + (1 / 3) * flow[
                                        i + j]
                                print(index, 'a')
                            elif closest_weekends == 1:
                                if closest.index[index] % 7 == 4 or closest.index[index] % 7 == 5:
                                    for j in range(flow_num):  # 若是则挨个取平均
                                        forecasting_flow.ix[index, j] = forecasting_flow.ix[index, j] + (1 / 3) * flow[
                                            i + j]
                                else:
                                    for j in range(flow_num):  # 若是则挨个取平均
                                        forecasting_flow.ix[index, j] = forecasting_flow.ix[index, j] + (1 / 3) * flow[
                                            i + j]
                                print(index, 'b')
                            elif closest_weekends == 2:
                                if closest.index[index] % 7 == 4 or closest.index[index] % 7 == 5:
                                    for j in range(flow_num):  # 若是则挨个取平均
                                        forecasting_flow.ix[index, j] = forecasting_flow.ix[index, j] + (1 / 3) * flow[
                                            i + j]
                                else:
                                    for j in range(flow_num):  # 若是则挨个取平均
                                        forecasting_flow.ix[index, j] = forecasting_flow.ix[index, j] + (1 / 3) * flow[
                                            i + j]
                                print(index, 'c')
                        else:
                            if closest_weekends == 0:
                                for j in range(flow_num):  # 若是则挨个取平均
                                    forecasting_flow.ix[index, j] = forecasting_flow.ix[index, j] + (1 / 3) * flow[
                                        i + j]
                                print(index, 'e')
                            elif closest_weekends == 1:
                                if closest.index[index] % 7 == 4 or closest.index[index] % 7 == 5:
                                    for j in range(flow_num):  # 若是则挨个取平均
                                        forecasting_flow.ix[index, j] = forecasting_flow.ix[index, j] + (1 / 3) * flow[
                                            i + j]
                                else:
                                    for j in range(flow_num):  # 若是则挨个取平均
                                        forecasting_flow.ix[index, j] = forecasting_flow.ix[index, j] + (1 / 3) * flow[
                                            i + j]
                                print(index, 'f')
                            elif closest_weekends == 2:
                                if closest.index[index] % 7 == 4 or closest.index[index] % 7 == 5:
                                    for j in range(flow_num):  # 若是则挨个取平均
                                        forecasting_flow.ix[index, j] = forecasting_flow.ix[index, j] + (1 / 3) * flow[
                                            i + j]
                                else:
                                    for j in range(flow_num):  # 若是则挨个取平均
                                        forecasting_flow.ix[index, j] = forecasting_flow.ix[index, j] + (1 / 3) * flow[
                                            i + j]
                                print(index, 'g')
                        forecasting_flow.iloc[index] = forecasting_flow.iloc[index] + (
                                    sum(flow[((aim_day + index) * flow_num): (aim_day + index + 1) * flow_num]) - sum(
                                forecasting_flow.iloc[index])) / flow_num
                        # forecasting_flow.iloc[index] = forecasting_flow.iloc[index] + (cum_flow.iloc[aim_day + index] - sum(forecasting_flow.iloc[index])) / flow_num
                        print(forecasting_flow)
                    count += 1  # 天计数器加一

        # print(cum_flow.iloc[aim_day], 'and', sum(flow[(aim_day * 333): (aim_day + 1) * 333]))
        print('预测值为：\n{}'.format(forecasting_flow))

    def weighted_markov_chain(self, error):
        # 对当前日期前两周的误差情况进行统计，得到几种状态对应的概率转移矩阵
        """
        prob_look_back = 7
        prob = []
        error.tolist()
        for i in range(len(error) - prob_look_back):
            a = error[i: (i + prob_look_back)]
            prob.append(a)
        """
        # ! 存在问题，到底是做离线还是在线的概率转移矩阵？离线可以用train_set做，在线则牵扯到迭代的问题，且需要屏蔽前14天的数值，且效果不一定好。
        # 离线方案可以首选，和time series的shape distance一样。
        # 计算加权马尔科夫链权重系数（层出错，主要还是在计算各阶自相关系数上）
        mean = np.mean(error)
        grade = 5  # 5阶（级）
        numerater = np.zeros([len(error) - 1, grade])  # 分子只有1~（n-k）项，此处取最大值
        denomiter = np.zeros(len(error))

        r = np.zeros(grade)
        w = np.zeros(grade)
        # 先计算sum((xi - x_mean)2)
        for j in range(len(error)):
            denomiter[j] = np.math.pow((error[j] - mean), 2)
        for i in range(1, grade + 1):  # (1-5)i用以表示阶数尺度，j用以表示error长度尺度
            for j in range(len(error) - i):
                numerater[j, i - 1] = (error[j] - mean) * (error[j + i] - mean)
        # print('分子与分母分别为：\n{}, \n{}'.format(numerater, denomiter))
        for i in range(grade):
            r[i] = sum(numerater[:, i]) / sum(denomiter)
        # print('各阶自相关系数为：{}'.format(r))
        # 各阶自相关系数归一化（确保之后加权求得的值不会大于1）
        r = abs(r)  # 求权重时，自相关系数取绝对值
        for i in range(grade):
            w[i] = r[i] / sum(r)
        # print('权重系数为：{}'.format(w))

        state = np.zeros(len(error))
        # 状态划分
        for i in range(len(error)):
            if error[i] <= -0.012:
                state[i] = -2
            elif -0.012 < error[i] < -0.005:
                state[i] = -1
            elif -0.005 <= error[i] <= 0.005:
                state[i] = 0
            elif 0.005 < error[i] < 0.012:
                state[i] = 1
            elif error[i] >= 0.012:
                state[i] = 2

        # 计算加权马尔可夫转移矩阵
        count = np.zeros([grade, grade, grade])  # 第三个维度表示各阶自相关系数所对应的概率转移矩阵
        # 初步统计转移数目
        for i in range(1, grade + 1):  # 与之前一样，大循环看各阶的值
            for j in range(len(error) - i):  # i表示马尔科夫阶数，j表示从当前时刻开始进行转移，j+i即表示从当前时刻转移i步对应的状态转移情况
                if state[j] == -2:
                    if state[j + i] == -2:
                        count[i - 1, 0, 0] += 1
                    elif state[j + i] == -1:
                        count[i - 1, 0, 1] += 1
                    elif state[j + i] == 0:
                        count[i - 1, 0, 2] += 1
                    elif state[j + i] == 1:
                        count[i - 1, 0, 3] += 1
                    elif state[j + i] == 2:
                        count[i - 1, 0, 4] += 1
                elif state[j] == -1:
                    if state[j + i] == -2:
                        count[i - 1, 1, 0] += 1
                    elif state[j + i] == -1:
                        count[i - 1, 1, 1] += 1
                    elif state[j + i] == 0:
                        count[i - 1, 1, 2] += 1
                    elif state[j + i] == 1:
                        count[i - 1, 1, 3] += 1
                    elif state[j + i] == 2:
                        count[i - 1, 1, 4] += 1
                elif state[j] == 0:
                    if state[j + i] == -2:
                        count[i - 1, 2, 0] += 1
                    elif state[j + i] == -1:
                        count[i - 1, 2, 1] += 1
                    elif state[j + i] == 0:
                        count[i - 1, 2, 2] += 1
                    elif state[j + i] == 1:
                        count[i - 1, 2, 3] += 1
                    elif state[j + i] == 2:
                        count[i - 1, 2, 4] += 1
                elif state[j] == 1:
                    if state[j + i] == -2:
                        count[i - 1, 3, 0] += 1
                    elif state[j + i] == -1:
                        count[i - 1, 3, 1] += 1
                    elif state[j + i] == 0:
                        count[i - 1, 3, 2] += 1
                    elif state[j + i] == 1:
                        count[i - 1, 3, 3] += 1
                    elif state[j + i] == 2:
                        count[i - 1, 3, 4] += 1
                elif state[j] == 2:
                    if state[j + i] == -2:
                        count[i - 1, 4, 0] += 1
                    elif state[j + i] == -1:
                        count[i - 1, 4, 1] += 1
                    elif state[j + i] == 0:
                        count[i - 1, 4, 2] += 1
                    elif state[j + i] == 1:
                        count[i - 1, 4, 3] += 1
                    elif state[j + i] == 2:
                        count[i - 1, 4, 4] += 1
        # print('状态统计结果为：\n{}\n'.format(count))
        # 计算转移概率
        prob = np.zeros([grade, grade, grade])
        distribute_prob = np.zeros([grade, grade])
        weighted_prob = np.zeros(grade)
        # 计算各阶状态转移概率矩阵
        for m in range(grade):
            for i in range(grade):
                if sum(count[m, i, :]) != 0:  # 排除分母为0的可能性
                    for j in range(grade):
                        prob[m, i, j] = count[m, i, j] / sum(count[m, i, :])
        # print('状态转移概率计算结果为：\n{}\n'.format(prob))
        # 计算加权后的状态转移概率矩阵
        for m in range(1, grade + 1):  # 从步长为1开始算
            # print(state[len(error) - m])
            if state[len(error) - m] == -2:
                distribute_prob[m - 1, :] = prob[m - 1, 0, :]
            elif state[len(error) - m] == -1:
                distribute_prob[m - 1, :] = prob[m - 1, 1, :]
            elif state[len(error) - m] == 0:
                distribute_prob[m - 1, :] = prob[m - 1, 2, :]
            elif state[len(error) - m] == 1:
                distribute_prob[m - 1, :] = prob[m - 1, 3, :]
            elif state[len(error) - m] == 2:
                distribute_prob[m - 1, :] = prob[m - 1, 4, :]
        # print(distribute_prob)
        for i in range(grade):
            for j in range(grade):
                weighted_prob[i] = weighted_prob[i] + (w[i] * distribute_prob[j, i])
        # print('加权状态转移概率为：\n{}\n'.format(weighted_prob))
        index = np.argmax(weighted_prob)
        cor_error = 0
        if index == 0:
            cor_error = -0.02
        elif index == 1:
            cor_error = -0.008
        elif index == 2:
            cor_error = 0.00
        elif index == 3:
            cor_error = 0.008
        elif index == 4:
            cor_error = 0.02
        return cor_error

class lstm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(lstm, self).__init__()
        self.reg = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, t = self.rnn(x)
        seq_len, batch_size, input_dim = x.shape
        x = x.view(seq_len * batch_size, input_dim)
        x = self.reg(x)
        x = x.view(seq_len, batch_size, -1)
        return x









