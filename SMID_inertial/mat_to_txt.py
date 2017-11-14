import os
import pandas as pd
import numpy as np
from scipy.io import loadmat
from sklearn import preprocessing

TRAIN_FILE_DIR = 'TrainInertial'
TEST_FILE_DIR = 'TestInertial'

INPUT_SIGNAL_TYPES = [
        'acc_x_',
        'acc_y_',
        'acc_z_',
        'gyro_x_',
        'gyro_y_',
        'gyro_z_',
    ]

TXT = 'TXT/'

def convert_mat_to_train_txt():
    acc_x_list = []
    acc_y_list = []
    acc_z_list = []
    gyro_x_list = []
    gyro_y_list = []
    gyro_z_list = []
    category_list = []
    for file in os.listdir(TRAIN_FILE_DIR):
        if not file.startswith("."):
            category = str(file.split('_')[0][1:]) + '\n' 
            category_list.append(category)
            data = loadmat(TRAIN_FILE_DIR + '/' + file)
            df = pd.DataFrame(data['d_iner'])
            x = df.values #returns a numpy array
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            df = pd.DataFrame(x_scaled)
            df_transpose = df.transpose()            
            df_transpose.to_csv('temp.csv', mode='w', index=False, sep=',', header=False)
            with open('temp.csv', 'r') as ro:
                temp_list = ro.readlines()
                acc_x_list.append(temp_list[0])
                acc_y_list.append(temp_list[1])
                acc_z_list.append(temp_list[2])
                gyro_x_list.append(temp_list[3])
                gyro_y_list.append(temp_list[4])
                gyro_z_list.append(temp_list[5])
    with open(TXT + 'y_train.txt', 'w') as wc:
        wc.writelines(category_list)
    for signal in INPUT_SIGNAL_TYPES:
        filename = TXT + signal + 'train.txt'
        print(filename)
        with open(filename, 'w') as wo:
            wo.writelines(eval(signal + 'list'))
        with open(filename, 'r') as r:
            sensor_data = r.read().split('\n')[:-1]
            data_list = []
            length_list = []
            for s_data in sensor_data:
                if s_data.strip() != None:
                    data_list.append(s_data.split(','))
                    length_list.append(len(s_data.split(',')))
            max_len = sorted(length_list, reverse=True)[0]
            string_list = []
            for d_element in data_list:
                diff = max_len - len(d_element)
                if diff != 0:
                    for i in range(diff):
                        d_element.append(d_element[-1])
                string_list.append(','.join(str(x) for x in d_element))
        with open(filename, 'w') as rw:
            for s_element in string_list:
                rw.write(s_element + '\n')

def convert_mat_to_test_txt():
    acc_x_list = []
    acc_y_list = []
    acc_z_list = []
    gyro_x_list = []
    gyro_y_list = []
    gyro_z_list = []
    category_list = []
    for file in os.listdir(TRAIN_FILE_DIR):
        if not file.startswith("."):
            data = loadmat(TRAIN_FILE_DIR + '/' + file)
            df = pd.DataFrame(data['d_iner'])
            x = df.values #returns a numpy array
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            df = pd.DataFrame(x_scaled)
            df_transpose = df.transpose()
            category = str(file.split('_')[0][1:]) + '\n' 
            category_list.append(category)
            df_transpose.to_csv('temp.csv', mode='w', index=False, sep=',', header=False)
            with open('temp.csv', 'r') as ro:
                temp_list = ro.readlines()
                acc_x_list.append(temp_list[0])
                acc_y_list.append(temp_list[1])
                acc_z_list.append(temp_list[2])
                gyro_x_list.append(temp_list[3])
                gyro_y_list.append(temp_list[4])
                gyro_z_list.append(temp_list[5])
    with open(TXT + 'y_test.txt', 'w') as wc:
        wc.writelines(category_list)
    for signal in INPUT_SIGNAL_TYPES:
        filename = TXT + signal + 'test.txt'
        print(filename)
        with open(filename, 'w') as wo:
            wo.writelines(eval(signal + 'list'))
        with open(filename, 'r') as r:
            sensor_data = r.read().split('\n')[:-1]
            data_list = []
            length_list = []
            for s_data in sensor_data:
                if s_data.strip() != None:
                    data_list.append(s_data.split(','))
                    length_list.append(len(s_data.split(',')))
            max_len = sorted(length_list, reverse=True)[0]
            string_list = []
            for d_element in data_list:
                diff = max_len - len(d_element)
                if diff != 0:
                    for i in range(diff):
                        d_element.append(d_element[-1])
                string_list.append(','.join(str(x) for x in d_element))
        with open(filename, 'w') as rw:
            for s_element in string_list:
                rw.write(s_element + '\n')
            
convert_mat_to_train_txt()
convert_mat_to_test_txt()
