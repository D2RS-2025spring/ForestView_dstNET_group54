import os
import numpy as np
import rasterio
import tqdm

# 输入：文件夹路径列表 和 输出文件路径列表
input_folders = [


    r"H:\szbj\S2_keerqin\4",
    r"H:\szbj\S2_keerqin\5",
    r'H:\szbj\S2_keerqin\6\1',
    r'H:\szbj\S2_keerqin\6\2',
    r'H:\szbj\S2_keerqin\7\1',
    r'H:\szbj\S2_keerqin\7\2',
    r"H:\szbj\S2_keerqin\8",
    r"H:\szbj\S2_keerqin\9",
    r"H:\szbj\S2_keerqin\10",
    r'H:\szbj\S2_keerqin\11\1',
    r'H:\szbj\S2_keerqin\11\2',
    r'H:\szbj\S2_keerqin\11\3',
    r'H:\szbj\S2_keerqin\11\4',
    r'H:\szbj\S2_keerqin\12\1',
    r'H:\szbj\S2_keerqin\12\2',
    r'H:\szbj\S2_keerqin\12\3',
    r'H:\szbj\S2_keerqin\12\4',
    r"H:\szbj\S2_keerqin\13"
    # 你可以继续添加更多路径
]

output_files = [


    r"H:\szbj\S2_keerqin\4_21.tif",
    r"H:\szbj\S2_keerqin\5_21.tif",
    r'H:\szbj\S2_keerqin\6\6_1_21.tif',
    r'H:\szbj\S2_keerqin\6\6_2_21.tif',
    r'H:\szbj\S2_keerqin\7\7_1_21.tif',
    r'H:\szbj\S2_keerqin\7\7_2_21.tif',
    r"H:\szbj\S2_keerqin\8_21.tif",
    r"H:\szbj\S2_keerqin\9_21.tif",
    r"H:\szbj\S2_keerqin\10_21.tif",
    r'H:\szbj\S2_keerqin\11\11_1_21.tif',
    r'H:\szbj\S2_keerqin\11\11_2_21.tif',
    r'H:\szbj\S2_keerqin\11\11_3_21.tif',
    r'H:\szbj\S2_keerqin\11\11_4_21.tif',
    r'H:\szbj\S2_keerqin\12\12_1_21.tif',
    r'H:\szbj\S2_keerqin\12\12_2_21.tif',
    r'H:\szbj\S2_keerqin\12\12_3_21.tif',
    r'H:\szbj\S2_keerqin\12\12_4_21.tif',
    r"H:\szbj\S2_keerqin\13_21.tif"
    # 和上面的路径一一对应
]

# 遍历每对输入文件夹和输出路径
for folder_path, output_path in zip(input_folders, output_files):
    print(f'处理文件夹：{folder_path}')

    image_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    image_files.sort()

    band1_stack = []
    band2_stack = []
    band3_stack = []
    band4_stack = []

    # 获取影像尺寸和空间信息
    with rasterio.open(os.path.join(folder_path, image_files[0])) as src:
        crs = src.crs
        transform = src.transform
        height = src.height
        width = src.width

    for image_file in tqdm.tqdm(image_files, desc='读取波段数据'):
        image_path = os.path.join(folder_path, image_file)
        with rasterio.open(image_path) as src:
            band1_stack.append(src.read(1))
            band2_stack.append(src.read(2))
            band3_stack.append(src.read(3))
            band4_stack.append(src.read(4))

    band1_stack = np.dstack(band1_stack)
    band2_stack = np.dstack(band2_stack)
    band3_stack = np.dstack(band3_stack)
    band4_stack = np.dstack(band4_stack)

    final_stack = np.concatenate([band1_stack, band2_stack, band3_stack, band4_stack], axis=-1)

    with rasterio.open(output_path, 'w', driver='GTiff',
                       height=height, width=width,
                       count=final_stack.shape[2], dtype=final_stack.dtype,
                       crs=crs, transform=transform) as dst:
        for i in tqdm.tqdm(range(final_stack.shape[2]), desc='写入合成影像'):
            dst.write(final_stack[:, :, i], i + 1)

    print(f'合成的影像已保存为 {output_path}')
