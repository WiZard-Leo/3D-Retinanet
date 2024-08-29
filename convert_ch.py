import os
import numpy as np
from tqdm import tqdm

def remove_last_channel(folder_path):
    for file_name in tqdm(os.listdir(folder_path)):
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path, file_name)
            # 加载.npy文件
            data = np.load(file_path)
            
            # 检查数据形状并去除最后一个单通道
            if data.shape[-1] == 1:
                data = data[..., 0]  # 去除最后一个单通道
            
            # 保存修改后的数据
            np.save(os.path.join("/opt/data/private/lwz/workplace/tutorials/data/ctdata_npy_256/",file_name), data)
            # print(f"Processed {file_name}: new shape {data.shape}")

# 使用示例
folder_path = '/opt/data/private/lwz/workplace/tutorials/data/ctdata_npy/'  # 替换为你存放.npy文件的文件夹路径
remove_last_channel(folder_path)