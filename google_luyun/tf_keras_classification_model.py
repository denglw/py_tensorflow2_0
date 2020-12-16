#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/8/4 22:21
# @Author  : Denglw
# @Email   : 892253193@qq.com
# @File    : tf_keras_classification_model.py
# @desc: 分类问题   DNN + StandardScaler + BatchNormalization + Dropout + callbacks
# 数据集--> fashion_mnist 时尚衣服、鞋子、包   10分类

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import sys
import os
import time
import sklearn
import tensorflow as tf

from tensorflow import keras

print(tf.__version__)
print(sys.version_info)
for model in mpl, np, pd, sklearn, tf, keras:
    print(model.__name__, model.__version__)


# fashion_mnist 时尚衣服、鞋子、包
fashion_mnist = keras.datasets.fashion_mnist
(x_train_all,y_train_all),(x_test,y_test) = fashion_mnist.load_data()
x_valid,x_train = x_train_all[:10000], x_train_all[10000:]
y_valid,y_train = y_train_all[:10000], y_train_all[10000:]
print(x_valid.shape,y_valid.shape)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

print(np.max(x_train),np.min(x_train))

# 数据归一化 StandardScaler
# y = (x - u) / std
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# x_train: [None, 28, 28] -> [None, 784]  要求为二维数据 故而进行转换
x_train_scaled = scaler.fit_transform(x_train.astype(np.float32).reshape(-1,1)).reshape(-1,28,28) # fit_transform
x_valid_scaled = scaler.transform(x_valid.astype(np.float32).reshape(-1,1)).reshape(-1,28,28) # transform
x_test_scaled = scaler.transform(x_test.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)

print(np.max(x_train_scaled),np.min(x_train_scaled))

# tf.keras.models.Sequential()
# DNN 深度神经网络
# BatchNormalization 批归一化
# Dropout 解决过拟合
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
for i in range(10):
    model.add(keras.layers.Dense(100,activation='selu'))
    model.add(keras.layers.BatchNormalization())
    '''
    # 激活函数 之前 添加批归一化
    model.add(keras.layers.Dense(100))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    '''
#model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.AlphaDropout(rate=0.5))
# AlphaDropout 1、均值和方差不变，2、归一化性质不变
model.add(keras.layers.Dense(10,activation='softmax'))
# relu: y = max(0, x)
# softmax : 将向量变成概率分布， x = [x1, x2, x3]
# y = [e^x1/sum + e^x2/sum + e^x3/sum] =1, sum = e^x1 + e^x2 + e^x3

'''
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300,activation='relu'))
model.add(keras.layers.Dense(200,activation='relu'))
model.add(keras.layers.Dense(100,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))
'''

'''
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(300,activation='relu'),
    keras.layers.Dense(100,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])
'''

# sparse原因 y->index. y->one_hot
# sparse_categorical_crossentropy
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd' , metrics =['accuracy'])

model.layers
model.summary()
# 参数Param计算 全连接层Dense  例如 235500
#[None, 784] * W + b -> [None, 300]        W.shape[784,300] , b=[300]

# callbacks回调函数
# TensorBoard  ModelCheckpoint  EarlyStopping
logdir = os.path.join('dnn-selu-dropout-callbacks')
if not os.path.exists(logdir):
    os.mkdir(logdir)
out_put_modelfile = os.path.join(logdir,"fashion_mnist_model.h5")
callbacks = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(out_put_modelfile),
    keras.callbacks.EarlyStopping(min_delta=1e-5,patience=5)
]
history = model.fit(x_train_scaled,y_train,epochs=10,validation_data=(x_valid_scaled,y_valid),callbacks=callbacks)
# 异常处理  http://www.mamicode.com/info-detail-2889136.html
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

plot_learning_curves(history)

#评估
model.evaluate(x_test_scaled,y_test)
# tensorboard 展示，命令行下执行，注意：文件目录
# tensorboard --logdir=callbacks



