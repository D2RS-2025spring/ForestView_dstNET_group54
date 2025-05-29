# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 16:09:18 2022

@author: d
"""

from osgeo import gdal
import numpy as np
import glob

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
    #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans) #写入仿射变换参数
        dataset.SetProjection(im_proj) #写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset
# %%以下函数都是一些数据增强的函数
def rotate(x, y, angle):
    k=angle//90
    x = np.rot90(x,k,(1,2))
    y = np.rot90(y,k,(1,2))
    return x,y

def data_aug(x,y,index):
    """
        五种数据增广方式，旋转90、180、270，左右、上下翻转
    """
    if index == 0:
        x,y = rotate(x,y, 90)  
    elif index==1:
        x,y = rotate(x,y, 180)
    elif index==2:
        x,y = rotate(x,y, 270)

    elif index==3:  
        x = np.flip(x,2) 
        y = np.flip(y,2)     
    else:         
        x = np.flip(x,1) 
        y = np.flip(y,1)  
    return x,y

img_dst = glob.glob(r'G:\center pivot\tpslab\lstmunet\clip\train\img\*.tif')
lbl_dst = glob.glob(r'G:\center pivot\tpslab\lstmunet\clip\train\lab\*.tif')

num = 1
for i in range(len(img_dst)):
    

    img_data = readTif(img_dst[i])
    
    geotrans = img_data[4]
    proj = img_data[5]
    img_data = img_data[0]
    
    
    lbl_data = readTif(lbl_dst[i])
    lbl_dat1 = [lbl_data[0]]
    lbl_data = np.array(lbl_dat1)


    for j in range(5):
        tmp_img_data,tmp_lbl_data = data_aug(img_data,lbl_data,j)
        img_flname = r"G:\center pivot\tpslab\lstmunet\aug\train\img\\" + img_dst[i].split('\\')[-1] +"%02d.tif" % j
        lbl_flname = r'G:\center pivot\tpslab\lstmunet\aug\train\lab\\' + lbl_dst[i].split('\\')[-1] +"%02d.tif" % j
        
        writeTif(tmp_img_data, geotrans, proj, img_flname)
        writeTif(tmp_lbl_data, geotrans, proj, lbl_flname)
        num += 1
        