# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 15:45:41 2022

@author: PC



"""

import glob
import numpy as np
# from matplotlib import pyplot as plt
import random
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization,ConvLSTM2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Multiply, Conv2D, Concatenate, SeparableConv2D, Activation, Add, Conv2DTranspose
import tensorflow.keras.backend as K


import os
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from tensorflow.keras.callbacks import ModelCheckpoint
from osgeo import gdal

# 读取遥感影像，需要安装GDAL
def readTif(fileName, xoff = 0, yoff = 0, data_width = 0, data_height = 0):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    #  栅格矩阵的列数
    width = dataset.RasterXSize 
    #  栅格矩阵的行数
    height = dataset.RasterYSize 
    #  波段数
    bands = dataset.RasterCount 
    #  获取数据
    if(data_width == 0 and data_height == 0):
        data_width = width
        data_height = height
    data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)
    #  获取仿射矩阵信息
    geotrans = dataset.GetGeoTransform()
    #  获取投影信息
    proj = dataset.GetProjection()
    return data


def shape_attention(input_tensor):
    # 利用平均池化和最大池化捕捉全局统计信息，得到形状 (batch, H, W, 1)
    avg_pool = tf.reduce_mean(input_tensor, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(input_tensor, axis=-1, keepdims=True)

    #拼接池化结果，形状为 (batch, H, W, 2)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])

    # 使用空洞卷积来扩展感受野
    conv = SeparableConv2D(1, (3, 3), dilation_rate=(2, 2), padding="same",
                  activation="relu",
                  depthwise_initializer="he_normal",
                  pointwise_initializer="he_normal")(concat)

    # 通过多层卷积提取更细粒度的空间信息
    conv2 = SeparableConv2D(1, (3, 3), padding="same",
                   activation="relu",
                   depthwise_initializer="he_normal",
                   pointwise_initializer="he_normal")(conv)

    # 5. 通过1×1卷积整合为单通道注意力图，形状为 (batch, H, W, 1)
    attention = Conv2D(1, (1, 1), padding="same",
                       activation="sigmoid",
                       kernel_initializer="he_normal")(conv2)

    # 6. 利用注意力图对原始特征进行加权
    return Multiply()([input_tensor, attention])


# 定义残差块
def residual_block(x, filters, kernel_size=3):
    shortcut = Conv2D(filters, (1,1), padding="same", kernel_initializer="he_normal")(x)  # 1×1 卷积匹配通道数
  # 残差连接的输入

    x = Conv2D(filters, kernel_size, padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, kernel_size, padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)

    # 添加残差连接（输入 + 卷积输出）
    x = Add()([shortcut, x])
    x = Activation("relu")(x)

    return x


def focal_loss(alpha=0.25, gamma=1.5):
    """
    Focal Loss 实现（二分类）

    参数：
        alpha (float): 平衡正负样本的权重，默认 0.25
        gamma (float): 抑制简单样本的聚焦参数，默认 2.0
    """

    def focal_loss_fixed(y_true, y_pred):
        # 确保数值稳定性（防止 log(0)）
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

        # 计算交叉熵
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)

        # 计算 p_t（真实类别的概率）
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)

        # 计算调制因子 (1 - p_t)^gamma
        modulating_factor = K.pow(1 - p_t, gamma)

        # 应用 alpha 权重（正负样本不同）
        alpha_weight = y_true * alpha + (1 - y_true) * (1 - alpha)

        # 组合所有因子
        loss = alpha_weight * modulating_factor * cross_entropy

        # 返回均值损失
        return K.mean(loss, axis=-1)

    return focal_loss_fixed

# keras函数式建模
def unet(batch_size=4 ,input_size = (256,256,24), classNum = 1, learning_rate = 1e-4):
    inputs = Input(input_size) # 输入的图像大小（行，列，波段数）
    BS = batch_size
    def reshapes(embed):
        # => [BS, 256, 256, 4, 6]
        embed = tf.reshape(embed, [BS, 256, 256, 4, 6])
        # => [BS,  256, 256, 6, 4]
        embed = tf.transpose(embed, [0, 4, 1, 2, 3])
        # => [BS*256*256, 6, 4]
        #embed = tf.reshape(embed, [BS*256*256, 6, 4])
        return embed
    # => [BS*256*256, 6, 4]
    inputs1 = keras.layers.Lambda(reshapes)(inputs)
    conv_lstm1 = ConvLSTM2D(filters=64,
                            kernel_size=(3, 3),
                            padding="same",
                            return_sequences=False)(inputs1)

    #  2D卷积层
    conv1 = residual_block(conv_lstm1, 64)
    #  对于空间数据的最大池化
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = residual_block(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = residual_block(pool2, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = residual_block(pool3, 512)
    #  Dropout正规化，防止过拟合

    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = residual_block(pool4, 1024)
    drop5 = Dropout(0.5)(conv5)
    #  上采样之后再进行卷积，相当于转置卷积操作,在每个跳跃连接前添加注意力机制
    drop4_att = shape_attention(drop4)  # 对编码器特征加注意力
    up6 = Conv2D(512, 2, activation='relu', padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4_att, up6], axis=3)
    conv6 = BatchNormalization()(Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6))
    conv6 = BatchNormalization()(Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6))


    #conv3_att = shape_attention(conv3)  # 第二个跳跃连接
    up7 = Conv2D(256, 2, activation='relu', padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = BatchNormalization()(Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7))
    conv7 = BatchNormalization()(Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7))

    #conv2_att = shape_attention(conv2)  # 第三个跳跃连接
    up8 = Conv2D(128, 2, activation='relu', padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = BatchNormalization()(
        Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8))
    conv8 = BatchNormalization()(
        Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8))

    conv1_att = shape_attention(conv1)  # 第四个跳跃连接
    up9 = Conv2D(64, 2, activation='relu', padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1_att, up9], axis=3)
    conv9 = BatchNormalization()(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9))
    conv9 = BatchNormalization()(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9))
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)


    conv10 = Conv2D(classNum, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)
    #  用于配置训练模型（优化器、目标函数、模型评估标准）
    model.compile(
                 optimizer = Adam(lr = learning_rate),
                 loss = focal_loss(alpha=0.25, gamma=1.5), # Dice损失
                 metrics = ['Precision','Recall']
                 )

    return model


def train_generator(img_file_dst,lbl_file_dst,batch_size,shuffle):

    idx = np.arange(len(img_file_dst))
    
    if shuffle:
        np.random.shuffle(idx)
    total_data_num = len(img_file_dst)
    while(True):
        for i in range(total_data_num//batch_size):

            max_ = min((i+1)*batch_size,total_data_num)
            tmp_file = idx[i*batch_size:max_]

            img_data = np.ndarray([max_-i*batch_size,256,256,24])
            lbl_data = np.zeros([max_-i*batch_size,256,256,1])

            for j,tmp_idx in enumerate(tmp_file):

                tmp_img = readTif(img_file_dst[tmp_idx])
                tmp_lbl = readTif(lbl_file_dst[tmp_idx])
              
                img_data[j] = tmp_img.transpose(1,2,0)
                lbl_data[j,:,:,0] = tmp_lbl.squeeze()

            yield (img_data,lbl_data)

if __name__ =='__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    from tensorflow.python.keras.utils.vis_utils import plot_model
    import pydotplus
    batch_size = 4
    model = unet()
    plot_model(model, to_file='conlstmreat.png',show_shapes=True)

    train_image_dst = glob.glob(r'G:\center pivot\etklab\cpi训练\aug\train\img\*.tif')
    train_label_dst = glob.glob(r'G:\center pivot\etklab\cpi训练\aug\train\lab\*.tif')
    shuffle = True
    train_Generator = train_generator(
                                  train_image_dst,
                                  train_label_dst,
                                  batch_size,
                                  shuffle)

    val_image_dst = glob.glob(r'G:\center pivot\etklab\cpi训练\aug\test\img\*.tif')
    val_label_dst = glob.glob(r'G:\center pivot\etklab\cpi训练\aug\test\lab\*.tif')
    val_Generator = train_generator(
                                  val_image_dst,
                                  val_label_dst,
                                  batch_size,
                                  shuffle)

    steps_per_epoch = len(train_image_dst)/ batch_size
    val_steps = len(val_image_dst)/batch_size
    model_path =r"G:\center pivot\etklab\cpi训练\conlstmreunetat\4_fa_gamma1.5\lstmunet_model5.hdf5" # 模型存储地址
    model_checkpoint = ModelCheckpoint(model_path,
                                    monitor = 'loss',
                                    verbose = 1,# 日志显示模式:0->安静模式,1->进度条,2->每轮一行
                                    save_best_only = True)
    # 开始训练

    history=model.fit_generator(train_Generator,
                    steps_per_epoch = steps_per_epoch,
                    epochs =25,
                    callbacks = [model_checkpoint],  # 实时保存最佳模型
                    validation_data = val_Generator,
                    validation_steps = val_steps
                    )
    model.save(r'G:\center pivot\etklab\cpi训练\conlstmreunetat\4_fa_gamma1.5\lstmunet_model_v5.h5') # 保存最终模型
import pandas as pd
import matplotlib.pyplot as plt
pd.DataFrame(history.history).to_csv(r'G:\center pivot\etklab\cpi训练\conlstmreunetat\4_fa_gamma1.5\training_log5.csv', index=False)
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot (history.history['Precision'])
plt.plot (history.history['val_Precision'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2,1,2)
plt.plot (history. history['loss'])
plt.plot (history. history['val_loss'])
plt.title(' Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper right')
plt.tight_layout()
plt.show ()

    # %% 加载模型
#model = unet()
#model.load_weights(r'D:\learning\bssj\model_unet_v1.h5')





