"""
开发版本：v1
开发作者：吴舜禹
开发时间：2021.3.5
库功能： 1.预测数据集整理（用前look back步，预测后一步）
        2.评价指标集合
        3.通讯模块（Kafka通讯）
        4.可视化模块（路径导入画图、数据导入画图、部分强调画图）
"""

import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from kafka import KafkaProducer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from matplotlib.patches import ConnectionPatch
from datetime import datetime

def forecasting_dataset_perparation(data, look_back):
    """
    预测数据集准备
    :param data: 源数据
    :param look_back: 用作特征数据的前look back个数据
    :return: X_train: 训练集输入； Y_train: 训练集输出； X_test: 测试机输入； y_test: 测试机输出
    注：数据集整理格式为(seq_len, batch, input_size)；
        本函数所得数据集在训练时采用的方式full batch learning，该种方式对较小样本集有着更好的效果，可以更好的收敛。
        需要注意的是，该种数据集处理方法并不适用于在线过程，在线预测数据集的输入需要迭代更新，而不能使用full batch learning。
    """
    #设置数据集，即前look_back个时间序列用来预测后一个时间序列
    x_data, y_data = [], []
    data = data.tolist()
    for i in range(len(data) - look_back):
        a = data[i:(i + look_back)]
        x_data.append(a)
        y_data.append(data[i + look_back])
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    # 取前80%作为训练集，后20%为测试集，batch size为1
    trainset_size = int(len(x_data) * 0.8)
    x_train = x_data[:trainset_size]
    x_train = x_train.reshape(-1, 1, look_back)
    y_train = y_data[:trainset_size]
    y_train = y_train.reshape(-1, 1, 1)
    x_test = x_data[trainset_size:]
    x_test = x_test.reshape(-1, 1, look_back)
    y_test = y_data[trainset_size:]
    # 将训练集中的输入输出坐标数据转换成torch中的tensor
    X_train = torch.from_numpy(x_train)
    X_train = torch.tensor(X_train, dtype=torch.float32)  # 注，这里需要进行强制类型转换，否则会报错
    Y_train = torch.from_numpy(y_train)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_test = torch.from_numpy(x_test)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    return X_train, Y_train, X_test, y_test

class criteria:
    #评价指标
    def __init__(self, true, pred):
        """
        评价指标初始化
        :param true: 真实值
        :param pred: 预测值
        """
        self.true_value = true
        self.pred_value = pred

    def MAPE(self):
        #MAPE
        true = np.array(self.true_value)
        pred = np.array(self.pred_value)
        for i in range(len(true)):
            if true[i] == 0:
                true[i] = 0.01
        diff = np.abs(np.array(true) - np.array(pred))
        mape = np.mean(diff / true) * 100
        return mape
    
    def MAE(self):
        #MAE
        return mean_absolute_error(self.true_value, self.pred_value)

    def MSE(self):
        #MSE
        return mean_squared_error(self.true_value, self.pred_value)

    def LCAIR(self):
        #LCAIR，用以衡量预测滞后性，已投Neurocomputing
        lcair = (mean_squared_error(self.true_value[: len(self.true_value) - 1], self.pred_value[1: ]) - mean_squared_error(self.true_value, self.pred_value)) / mean_squared_error(self.true_value, self.pred_value)
        return lcair

class communication:
    #通讯模块
    def kafka_send(self, msg_dict, brokers, topic):
        """
        kafka发送信息
        :param msg_dict: 日志消息
        :param brokers: kafka实例名称
        :param topic: 主题
        :return: 无
        """
        #return
        msg_dict['calculate_time'] = str(datetime.now())[:19]
        msg = json.dumps(msg_dict)
        #brokers = "10.200.197.81:9092"
        #topic = "jd_data"
        conf = {'bootstrap.servers': brokers}
        p = KafkaProducer(**conf)
        try:
            # Produce line (without newline)
            p.send(topic, msg, partition=0)
        except BufferError:
            sys.stderr.write('%% Local producer queue is full (%d messages awaiting delivery): try again\n' %
                             len(p))
        #p.poll(0)
        sys.stderr.write('%% Waiting for %d deliveries\n' % len(p))
        p.flush()

class visualization:
    #可视化
    def date_plot_with_path(self, path, plot_date):
        """
        根据给定路径文件和给定目标日期画图
        :param path: 目标路径文件
        :param plot_date: 目标日期
        :return: 无
        """
        data = pd.read_csv(path, engine='python')
        # data = pd.read_excel(path)
        date = data['TIME']
        value = data['VALUE']
        date = pd.to_datetime(date)
        date = date.dt.strftime('%Y-%m-%d')  # 格式转换，消去小数。

        count = 0
        figure_num = len(plot_date)  # 要画这么多张图
        value_plot = pd.DataFrame(np.zeros([len(plot_date), figure_num]))
        plt.figure(0)
        for i in range(figure_num):
            for j in range(len(date)):  # 将选定日期对应的value数据存储
                if date[j] == plot_date[i]:  # 若为想要画的日期
                    value_plot.ix[count, i] = value.iloc[j]
                    count += 1
            count = 0
            plt.plot(value_plot[i], linewidth=3, alpha=0.6)
            plt.tight_layout()
            plt.title('Flow curve of day {}'.format(plot_date[i]))

    def date_plot_with_value(self, data, plot_date):
        """
        根据给定数据给给定目标日期画图
        :param data: 给定数据
        :param plot_date: 目标日期
        :return: 无
        """
        date = data['TIME']
        value = data['VALUE']
        date = pd.to_datetime(date)
        date = date.dt.strftime('%Y-%m-%d')  # 格式转换，消去小数。
        count = 0
        figure_num = len(plot_date)  # 要画这么多张图
        value_plot = pd.DataFrame(np.zeros([len(plot_date), figure_num]))
        plt.figure(0)
        for i in range(figure_num):
            for j in range(len(date)):  # 将选定日期对应的value数据存储
                if date[j] == plot_date[i]:  # 若为想要画的日期
                    value_plot.ix[count, i] = value.iloc[j]
                    count += 1
            count = 0
            plt.plot(value_plot[i], linewidth=3, alpha=0.6)
            plt.tight_layout()
            plt.title('Flow curve of day {}'.format(plot_date[i]))

    def local_highlight_plot(self, true, pred, left_top_down, left_top_up, left_top_l, left_top_r, right_top_down, right_top_up, right_top_l, right_top_r):
        """
        部分高亮画图
        :param true: 真实值
        :param pred: 预测值
        :param left_top_down: 左上角高亮数值下限
        :param left_top_up: 左上角高亮数值上限
        :param left_top_l: 左上角高亮横轴最小值
        :param left_top_r: 左上角高亮横轴最大值
        :param right_top_down: 右上角高亮数值下限
        :param right_top_up: 右上角高亮数值上限
        :param right_top_l: 右上角高亮横轴最小值
        :param right_top_r: 右上角高亮横轴最大值
        :return: 无
        注：部分参数还要修改，主要是下方大图的上下限以及connect path的位置
        """
        fig = plt.figure(figsize=(14, 6))
        plt.axes().get_yaxis().set_visible(False)
        plt.axes().get_xaxis().set_visible(False)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        # plt.tight_layout()
        plt.title('True value line and Forecasting value line', fontsize=18)
        # plt.subplots_adjust(bottom=0., left=0, top=1., right=1)
        # 创建第一个轴，左上角的图用绿色的图
        sub1 = fig.add_subplot(2, 2, 1)  # 两行两列，第一单元格
        sub1.plot(pred, color='xkcd:golden yellow', label="Corrected LSTM Forecasting",
                  linewidth=3)
        sub1.plot(true, color='xkcd:light red', label="True", linewidth=3)
        sub1.set_xlim(left_top_l, left_top_r)
        sub1.set_ylim(left_top_down, left_top_up)
        # sub1.set_ylim(0.2, .5)
        # sub1.set_ylabel('y', labelpad=15)

        # 创建第二个轴，即左上角的橙色轴
        sub2 = fig.add_subplot(2, 2, 2)  # 两行两列，第二个单元格
        sub2.plot(pred, color='xkcd:golden yellow', label="Corrected LSTM Forecasting",
                  linewidth=3)
        sub2.plot(true, color='xkcd:light red', label="True", linewidth=3)
        sub2.set_xlim(right_top_l, right_top_r)
        sub2.set_ylim(right_top_down, right_top_up)
        # sub2.set_ylim(.4, 1)

        # 创建第三个轴，第三和第四个单元格的组合
        sub3 = fig.add_subplot(2, 2, (3, 4))  # 两行两列，合并第三和第四单元格
        sub3.plot(pred, color='xkcd:golden yellow', label="Corrected LSTM Forecasting",
                  linewidth=3)
        sub3.plot(true, color='xkcd:light red', label="True", linewidth=3)
        sub3.legend()
        # sub3.set_xlim(0, 6.5)
        sub3.set_ylim(580, 660)
        # sub3.set_xlabel(r'$\theta$ (rad)', labelpad=15)
        # sub3.set_ylabel('y', labelpad=15)

        # 在第三个轴中创建阻塞区域333333333
        sub3.fill_between((left_top_l, left_top_r), left_top_down, left_top_up, facecolor='xkcd:light lavendar',
                          alpha=0.4)  # 第一个轴的阻塞区域
        sub3.fill_between((right_top_l, right_top_r), right_top_down, right_top_up, facecolor='xkcd:light orange',
                          alpha=0.4)  # 第二轴的阻塞区域

        # 在左侧创建第一个轴的ConnectionPatch
        con1 = ConnectionPatch(xyA=(left_top_l, 600), coordsA=sub1.transData,
                               xyB=(left_top_l, 600), coordsB=sub3.transData, color='xkcd:light lavendar', alpha=0.6,
                               linewidth=3)
        # 添加到左侧
        fig.add_artist(con1)
        # 在右侧创建第一个轴的ConnectionPatch
        con2 = ConnectionPatch(xyA=(left_top_r, 630), coordsA=sub1.transData,
                               xyB=(left_top_r, 630), coordsB=sub3.transData, color='xkcd:light lavendar', alpha=0.6,
                               linewidth=3)
        # 添加到右侧
        fig.add_artist(con2)
        # 在左侧创建第二个轴的ConnectionPatch
        con3 = ConnectionPatch(xyA=(right_top_l, 600), coordsA=sub2.transData,
                               xyB=(right_top_l, 600), coordsB=sub3.transData, color='xkcd:light orange', alpha=0.6,
                               linewidth=3)
        # 添加到左侧
        fig.add_artist(con3)
        # 在右侧创建第二个轴的ConnectionPatch
        con4 = ConnectionPatch(xyA=(right_top_r, 615), coordsA=sub2.transData,
                               xyB=(right_top_r, 615), coordsB=sub3.transData, color='xkcd:light orange', alpha=0.6,
                               linewidth=3)
        # 添加到右侧
        fig.add_artist(con4)