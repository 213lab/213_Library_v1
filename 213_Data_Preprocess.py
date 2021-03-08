"""
开发版本：v1
开发作者：吴舜禹
开发时间：2021.3.2
库功能： 1.数据导入（从csv或excel）
        2.NAN与0值处理；
        2.数据标准化
        3.数据归一化；
        4.时标数据重复值消除；
        5.时标数据缺失值插补；
        6.噪声数据平滑处理；
        7.one-hot编码
"""

import numpy as np
import pandas as pd
import datetime
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from scipy.signal import savgol_filter


def data_load_from_csv(path):
    data = pd.read_csv(path, engine='python')
    processed_data = data_nanAzero_process(data)
    return processed_data

def data_load_from_excel(path):
    data = pd.read_excel(path, engine='python')
    processed_data = data_nanAzero_process(data)
    return processed_data

def data_nanAzero_process(data):
    data['VALUE'].replace(0,np.nan,inplace=True)
    processed_data = data['VALUE'].fillna(data['VALUE'].interpolate())
    return processed_data

def data_normalization(data):
    raw_data = data.dropna()  # 去掉na数据
    raw_data = raw_data.values
    '''注：数据归一化处理时，不可以用原矩阵重新赋值！！！！！否则np.min的值会改变！！！'''
    processed_data = np.empty(len(raw_data))
    for i in range(len(processed_data)):
        processed_data[i] = float(raw_data[i] - np.min(raw_data)) / float(np.max(raw_data) - np.min(raw_data))
    return processed_data

def data_standardlization(data):
    return StandardScaler().fit(data)

def date_repetition_process(data, date_field='TIME', value_field='VALUE'):
    date = data[date_field]
    value = data[value_field]
    # 重复值剔除
    rep_proc_value = pd.Series(np.zeros(len(date)), index=range(len(date)))
    rep_proc_date = pd.Series(np.zeros(len(date)), index=range(len(date)))
    count = 0
    for i in range(len(date) - 1):  # 先将所有的数据进行去重处理
        if date[date.index[i]] != date[date.index[i + 1]]:
            rep_proc_date[count] = date[date.index[i]]  # 这种用法记住了
            rep_proc_value[count] = value[value.index[i]]
            count += 1
        else:
            print('重复值为：{}'.format(date[date.index[i]]))
    rep_proc_date[count] = date[date.index[i]]  # 补上最后一天的值
    rep_proc_value[count] = value[value.index[i]]
    count = 0
    print('重复值已剔除')
    print(rep_proc_date, rep_proc_value)
    processed_data = pd.concat([rep_proc_date, rep_proc_value], axis=1)
    processed_data.rename(columns={'0': 'TIME', '1': 'VALUE'}, inplace=True)
    return processed_data

def date_missing_process(data, target_len, date_field='TIME', value_field='VALUE'):
    count = 0
    date = data[date_field]
    value = data[value_field]
    # 缺失值插补
    miss_proc_value = pd.Series(np.zeros(target_len))
    miss_proc_date = pd.Series(np.zeros(target_len))
    miss_proc_date = pd.to_datetime(miss_proc_date)
    for j in range(target_len):
        miss_proc_date[count] = date[j]
        miss_proc_value[count] = value[j]
        while (miss_proc_date[count] + datetime.timedelta(minutes=1)) != date[j + 1]:
            print('插补的值为：{}'.format(miss_proc_date[count] + datetime.timedelta(minutes=1)))
            count += 1
            miss_proc_date[count] = miss_proc_date[count - 1] + datetime.timedelta(minutes=1)
            miss_proc_value[count] = (value[j] + value[j + 1]) / 2
        count += 1
        if count >= target_len:
            break
    print('日期缺失值已插补')
    processed_data = pd.concat([miss_proc_date, miss_proc_value], axis=1)
    processed_data.rename(columns={'0': 'TIME', '1': 'VALUE'}, inplace=True)
    return processed_data

def sequence_smooth(data, window_size, order):
    value = data['VALUE']
    date = data['TIME']
    processed_value = savgol_filter(value, window_size, order)
    return processed_value

def data_export_to_csv(data, filename):
    if not os.path.exists(filename):
        fd = open(filename, mode='w')
        fd.close()
    try:
        pd.DataFrame(data).to_csv(filename)
        print('{}数据已存储！'.format(filename))
    except:
        print('写入错误，请检查！')

def one_hot(data):
    values = np.array(data)
    #整数编码
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    #二进制编码
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    #第一个的编码结果
    inverted = label_encoder.inverse_transform([np.argmax(onehot_encoded[0, :])])
    return inverted





