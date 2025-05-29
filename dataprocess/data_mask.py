import arcpy
import os
from arcpy import env
from arcpy.sa import *
import numpy as np

# Set environment settings
# shp所在的文件夹
env.workspace = r"G:\center pivot\训练数据制作\fish\val"
# 裁剪后文件输出的文件夹
output_path = r"G:\center pivot\训练数据制作\lab_data\val"
# Set local variables
shps = arcpy.ListFeatureClasses()
# 需要裁剪的栅格
InValueRaster = r"G:\center pivot\训练数据制作\lab\lab_zong.tif"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 这个可以忽略，为了看下当前有多少数据需要处理
count = 0
for shp in shps:
    count = count + 1
print(count)

num = 0
# 循环
for shp in shps:
    try:
        # 获取文件名并去掉后缀
        num = num + 1
        file_name = shp.split('.')[0]
        print(file_name)

        # Check out the ArcGIS Spatial Analyst extension license
        arcpy.CheckOutExtension("Spatial")
        print(file_name)

        # Execute ExtractByMask
        outExtractByMask = ExtractByMask(InValueRaster, shp)
        print(file_name)

        # 替换 NoData 值为 0
        outRaster = Con(IsNull(outExtractByMask), 0, outExtractByMask)

        # 将栅格转换为 NumPy 数组
        arr = arcpy.RasterToNumPyArray(outRaster)

        if np.shape(arr)==(2049,2049):
           arr = arr[:-1, :-1]
        # 删除最后一列
        #arr = arr[:, :-1]
        print(np.shape(arr))
        # 将数组转换回栅格
        newRaster = arcpy.NumPyArrayToRaster(arr, outRaster.extent.lowerLeft, outRaster.meanCellWidth,outRaster.meanCellHeight)

        # 保存输出
        newRaster.save(output_path + '/' + file_name + '_c.tif')

        # 看当前处理了多少……
        print(num * 1.0 / count)
    except Exception as e:
        print(f"Error processing {shp}: {e}")

print('finish!')
