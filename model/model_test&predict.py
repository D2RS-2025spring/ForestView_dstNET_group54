import glob
import os

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization,ConvLSTM2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Multiply, Conv2D, Concatenate, SeparableConv2D, Activation, Add, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from osgeo import gdal
import tensorflow.keras.backend as K
from tensorflow.keras import losses
import datetime
import math
import sys


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
    return data, width, height, bands, geotrans, proj

#  保存tif文件函数
def writeTif(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    print(im_bands)
    #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans) #写入仿射变换参数
        dataset.SetProjection(im_proj) #写入投影
         # 写入数组数据
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

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
def unet(batch_size=1 ,input_size = (256,256,24), classNum = 1, learning_rate = 1e-4):
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


    conv3_att = shape_attention(conv3)  # 第二个跳跃连接
    up7 = Conv2D(256, 2, activation='relu', padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3_att, up7], axis=3)
    conv7 = BatchNormalization()(Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7))
    conv7 = BatchNormalization()(Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7))

    conv2_att = shape_attention(conv2)  # 第三个跳跃连接
    up8 = Conv2D(128, 2, activation='relu', padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2_att, up8], axis=3)
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



#TifFolderPath = sys.argv[1]
#ModelPath = sys.argv[2]
#ResultFolderPath = sys.argv[3]
#area_perc = float(sys.argv[4])
if __name__ =='__main__':
#  记录测试消耗时
    print(1)
    test_dst = glob.glob(r'G:\临时\16年256数据\16\9\*.tif')
    ResultFolderPath = r"G:\center pivot\perxin\2016\9\1"
    if not os.path.exists(ResultFolderPath):
        os.makedirs(ResultFolderPath)
    model = unet()
    model.load_weights(r'G:\center pivot\etklab\cpi训练\conlstmreunetat\41_gamma1.5\lstmunet_model_v5.h5')
    num = 1
    for i in range(len(test_dst)):
        test_data = readTif(test_dst[i])

        geotrans = test_data[4]
        proj = test_data[5]
        test_data = test_data[0]
        test_data = test_data.transpose(1, 2, 0)

        test_data = [test_data]
        print(np.shape(test_data))
        if np.shape(test_data) == (1, 256, 256, 24):
            test_data = np.array(test_data)

            results = model.predict(test_data, verbose=1, batch_size=1)
            print(np.shape(results))
            results = np.round(results).astype(int)
            print(np.shape(results))
            results = results.transpose(3, 1, 2, 0)
            print(np.shape(results))
            results = np.squeeze(results, axis=-1)
            print(np.shape(results))

            test_flname = ResultFolderPath + test_dst[i].split('\\')[-1] + "%d.tif" % i
            writeTif(results, geotrans, proj, test_flname)
            print(num)
            num += 1


   
