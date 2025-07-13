import os
import gzip
import shutil

def decompress_gz_file(file_path, output_path):
    """解压.gz文件到指定的输出路径"""
    with gzip.open(file_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(file_path)  # 删除原始.gz文件

def walk_and_decompress(directory):
    """遍历目录，解压所有.gz文件"""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.gz'):
                file_path = os.path.join(root, file)
                output_path = os.path.join(root, os.path.splitext(file)[0])
                decompress_gz_file(file_path, output_path)
                print(f'Decompressed and removed: {file_path} -> {output_path}')

# 请替换下面的路径为你的目标目录路径
target_directory = None
walk_and_decompress(target_directory)
