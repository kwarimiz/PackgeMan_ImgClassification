import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from multiprocessing import Pool
from tqdm import tqdm

def process_and_save_image(args):
    img_data, index, save_path = args

    # 图像处理
    times = 0.02
    img_data[0] = img_data[0] * 12.50 * times
    img_data[1] = img_data[1] * 7.143 * times
    img_data[2] = img_data[2] * 5.263 * times
    img_data = np.clip(img_data, 0, 1)

    # 保存图像
    plt.imshow(img_data.transpose(1, 2, 0))
    plt.savefig(os.path.join(save_path, f'{index}.png'))
    plt.close()  # 关闭图形以释放内存

def main(fits_path, save_path):
    with h5py.File(fits_path, 'r') as new_file:
        # 直接通过数据集名称访问数据
        images = new_file['images']
        
        # 准备参数
        tasks = [(images[i], i, save_path) for i in range(0,300)]

        # 创建多进程池
        with Pool(processes=os.cpu_count()) as pool:
            for _ in tqdm(pool.imap(process_and_save_image, tasks), total=len(tasks)):
                pass  # 这里可以添加其他需要做的事情，当前仅用于更新进度条


if __name__ == "__main__":
    fits_path = '/mnt/storage-data3/wangcheng/data/test_data/fits_test/test.h5' # 替换为您的 FITS 文件路径
    save_path = '/mnt/storage-data3/wangcheng/dataset/fits_to_png'  # 替换为您的保存路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    main(fits_path, save_path)
