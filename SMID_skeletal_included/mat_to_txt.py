import os
import pandas as pd
import numpy as np
from scipy.io import loadmat
from sklearn import preprocessing

TRAIN_INERTIAL = 'RawData/InertialTrainData'
TEST_INERTIAL = 'RawData/InertialTestData'
TRAIN_SKELETAL = 'RawData/SkeletalTrainData'
TEST_SKELETAL = 'RawData/SkeletalTestData'

INPUT_INERTIAL_SIGNAL_TYPES = [
        'acc_x_',
        'acc_y_',
        'acc_z_',
        'gyro_x_',
        'gyro_y_',
        'gyro_z_',
    ]

INERTIAL = 'ProcessedData/InertialData/'
SKELETAL = 'ProcessedData/SkeletalData/'

max_len = 0
max_len_s = 0

def clean_up(mydir):
    filelist = [ f for f in os.listdir(mydir) if f.endswith('.txt') ]
    for f in filelist:
        os.remove(os.path.join(mydir, f))

def convert_skeletal_mat_to_test_txt():
    global max_len_s
    category_list = []
    joint_x_list = []
    joint_y_list = []
    joint_z_list = []
    for file in os.listdir(TEST_SKELETAL):
        if not file.startswith("."):
            category = str(file.split('_')[0][1:]) + '\n' 
            category_list.append(category)
            with open(SKELETAL + 'y_test.txt', 'w') as wc:
                wc.writelines(category_list)
            data = loadmat(TEST_SKELETAL + '/' + file)
            temp_x_list = []
            temp_y_list = []
            temp_z_list = []
            for i in range(len(data['d_skel'])):
                df = pd.DataFrame(data['d_skel'][i])
                x = df.values #returns a numpy array
                min_max_scaler = preprocessing.MinMaxScaler()
                x_scaled = min_max_scaler.fit_transform(x)
                df = pd.DataFrame(x_scaled)
                df.to_csv('temp_skeletal.csv', mode='w', index=False, sep=',', header=False)
                with open('temp_skeletal.csv', 'r') as sw:
                    temp_list = sw.readlines()
                    temp_x_list.append(temp_list[0])
                    temp_y_list.append(temp_list[1])
                    temp_z_list.append(temp_list[2])
            joint_x_list.append(temp_x_list)
            joint_y_list.append(temp_y_list)
            joint_z_list.append(temp_z_list)
    for val in ['_x_', '_y_', '_z_']:
        for j in range(len(eval('joint' + val + 'list'))):
            for k in range(len(eval('joint' + val + 'list[j]'))):
                with open(SKELETAL + 'joint_' + str(k+1) + val + 'test.txt', 'a') as jw:
                    jw.writelines(eval('joint' + val + 'list[j][k]'))
    for val in ['_x_', '_y_', '_z_']:
        for k in range(20):
            with open(SKELETAL + 'joint_' + str(k+1) + val + 'test.txt', 'r') as jr:
                sk_data = jr.read().split('\n')[:-1]
                data_list = []
                length_list = []
                for s_data in sk_data:
                    if s_data.strip() != None:
                        data_list.append(s_data.split(','))
                        length_list.append(len(s_data.split(',')))
                # max_len_s = sorted(length_list, reverse=True)[0]
                string_list = []
                for d_element in data_list:
                    diff = max_len_s - len(d_element)
                    if diff != 0:
                        for i in range(diff):
                            d_element.append(d_element[-1])
                    string_list.append(','.join(str(x) for x in d_element))
            with open(SKELETAL + 'joint_' + str(k+1) + val + 'test.txt', 'w') as jw:
                print('Processed data files: joint_' + str(k+1) + val + 'test.txt')
                for s_element in string_list:
                    jw.write(s_element + '\n')

def convert_skeletal_mat_to_train_txt():
    global max_len_s
    category_list = []
    joint_x_list = []
    joint_y_list = []
    joint_z_list = []
    for file in os.listdir(TRAIN_SKELETAL):
        if not file.startswith("."):
            category = str(file.split('_')[0][1:]) + '\n' 
            category_list.append(category)
            with open(SKELETAL + 'y_train.txt', 'w') as wc:
                wc.writelines(category_list)
            data = loadmat(TRAIN_SKELETAL + '/' + file)
            temp_x_list = []
            temp_y_list = []
            temp_z_list = []
            for i in range(len(data['d_skel'])):
                df = pd.DataFrame(data['d_skel'][i])
                x = df.values #returns a numpy array
                min_max_scaler = preprocessing.MinMaxScaler()
                x_scaled = min_max_scaler.fit_transform(x)
                df = pd.DataFrame(x_scaled)
                df.to_csv('temp_skeletal.csv', mode='w', index=False, sep=',', header=False)
                with open('temp_skeletal.csv', 'r') as sw:
                    temp_list = sw.readlines()
                    temp_x_list.append(temp_list[0])
                    temp_y_list.append(temp_list[1])
                    temp_z_list.append(temp_list[2])
            joint_x_list.append(temp_x_list)
            joint_y_list.append(temp_y_list)
            joint_z_list.append(temp_z_list)
    for val in ['_x_', '_y_', '_z_']:
        for j in range(len(eval('joint' + val + 'list'))):
            for k in range(len(eval('joint' + val + 'list[j]'))):
                with open(SKELETAL + 'joint_' + str(k+1) + val + 'train.txt', 'a') as jw:
                    jw.writelines(eval('joint' + val + 'list[j][k]'))
    for val in ['_x_', '_y_', '_z_']:
        for k in range(20):
            with open(SKELETAL + 'joint_' + str(k+1) + val + 'train.txt', 'r') as jr:
                sk_data = jr.read().split('\n')[:-1]
                data_list = []
                length_list = []
                for s_data in sk_data:
                    if s_data.strip() != None:
                        data_list.append(s_data.split(','))
                        length_list.append(len(s_data.split(',')))
                max_len_s = sorted(length_list, reverse=True)[0]
                string_list = []
                for d_element in data_list:
                    diff = max_len_s - len(d_element)
                    if diff != 0:
                        for i in range(diff):
                            d_element.append(d_element[-1])
                    string_list.append(','.join(str(x) for x in d_element))
            with open(SKELETAL + 'joint_' + str(k+1) + val + 'train.txt', 'w') as jw:
                print(SKELETAL + 'joint_' + str(k+1) + val + 'train.txt')
                for s_element in string_list:
                    jw.write(s_element + '\n')

def convert_inertial_mat_to_train_txt():
    global max_len
    acc_x_list = []
    acc_y_list = []
    acc_z_list = []
    gyro_x_list = []
    gyro_y_list = []
    gyro_z_list = []
    category_list = []
    for file in os.listdir(TRAIN_INERTIAL):
        if not file.startswith("."):
            category = str(file.split('_')[0][1:]) + '\n' 
            category_list.append(category)
            data = loadmat(TRAIN_INERTIAL + '/' + file)
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
    with open(INERTIAL + 'y_train.txt', 'w') as wc:
        wc.writelines(category_list)
    for signal in INPUT_INERTIAL_SIGNAL_TYPES:
        filename = INERTIAL + signal + 'train.txt'
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

def convert_inertial_mat_to_test_txt():
    global max_len
    acc_x_list = []
    acc_y_list = []
    acc_z_list = []
    gyro_x_list = []
    gyro_y_list = []
    gyro_z_list = []
    category_list = []
    for file in os.listdir(TEST_INERTIAL):
        if not file.startswith("."):
            data = loadmat(TEST_INERTIAL + '/' + file)
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
    with open(INERTIAL + 'y_test.txt', 'w') as wc:
        wc.writelines(category_list)
    for signal in INPUT_INERTIAL_SIGNAL_TYPES:
        filename = INERTIAL + signal + 'test.txt'
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
            # max_len = sorted(length_list, reverse=True)[0]
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


clean_up(INERTIAL)
# clean_up(SKELETAL)
convert_inertial_mat_to_train_txt()
convert_inertial_mat_to_test_txt()
# convert_skeletal_mat_to_train_txt()
# convert_skeletal_mat_to_test_txt()
