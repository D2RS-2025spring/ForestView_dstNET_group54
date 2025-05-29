# -*- coding: utf-8 -*-
from osgeo import gdal
import numpy as np
import os


def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset is None:
        print(f"{fileName} 文件无法打开")
        return None
    data = dataset.ReadAsArray()
    geotrans = dataset.GetGeoTransform()
    proj = dataset.GetProjection()
    return data, geotrans, proj, dataset


def writeTif(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 2:
        im_data = np.expand_dims(im_data, axis=0)  # 添加波段维度
    im_bands, im_height, im_width = im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)

    if dataset is not None:
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)
        for i in range(im_bands):
            band = dataset.GetRasterBand(i + 1)
            band.WriteArray(im_data[i])
            band.SetNoDataValue(0)  # 设定 0 为 NoData 值
    del dataset


def clip_and_save(input_folder, output_folder, target_size=256):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_list = [f for f in os.listdir(input_folder) if f.endswith('.tif')]

    for idx, file in enumerate(file_list):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, f"clipped_{idx + 1:03d}.tif")

        # 读取原始影像
        result = readTif(input_path)
        if result is None:
            continue
        data, geotrans, proj, dataset = result

        width = dataset.RasterXSize
        height = dataset.RasterYSize

        # 计算裁剪起始点（从左上角开始）
        xoff = 0
        yoff = 0

        # 确保裁剪尺寸不会超出影像范围
        crop_width = min(target_size, width)
        crop_height = min(target_size, height)

        # 使用 gdal.Translate 进行裁剪
        options = gdal.TranslateOptions(
            srcWin=[xoff, yoff, crop_width, crop_height],  # 起始点+裁剪尺寸
            noData=0  # 设定 NoData 值
        )
        clipped_ds = gdal.Translate(output_path, dataset, options=options)

        # 释放数据集
        del dataset
        del clipped_ds

        print(f"文件 {file} 裁剪完成，保存至 {output_path}")


# 输入与输出路径
input_folder = r'G:\center pivot\其他地区\qita'
output_folder = r'G:\center pivot\其他地区\256'

clip_and_save(input_folder, output_folder)
