import os
import random

# 设置数据目录路径
data_folder = "./turntable-cropped"

# 设置训练集和测试集比例
train_ratio = 0.8

# 设置存储文件路径的txt文件
train_txt_path = "train.txt"
val_txt_path = "val.txt"

# 初始化存储训练集和测试集文件路径的列表
train_files = []
val_files = []

# 创建类别到数字的映射字典
category_to_number = {}

# 遍历每个类别的文件夹并为类别分配数字
for i, category in enumerate(os.listdir(data_folder)):
    category_folder = os.path.join(data_folder, category)

    # 检查是否是文件夹
    if os.path.isdir(category_folder):
        category_to_number[category] = i

# 遍历每个类别的文件夹
for category in os.listdir(data_folder):
    category_folder = os.path.join(data_folder, category)

    # 检查是否是文件夹
    if os.path.isdir(category_folder):
        # 获取文件夹下所有png文件的路径
        image_files = [os.path.join(category_folder, file) for file in os.listdir(category_folder) if file.endswith(".jpg") or file.endswith("png")]

        # 随机打乱文件顺序
        random.shuffle(image_files)

        # 计算切分位置
        split_index = int(len(image_files) * train_ratio)

        # 将文件路径添加到训练集和测试集列表中
        for file in image_files[:split_index]:
            train_files.append((os.path.abspath(file), category_to_number[category]))

        for file in image_files[split_index:]:
            val_files.append((os.path.abspath(file), category_to_number[category]))

# 打印训练集和测试集的样本数量
print(f"训练集样本数量: {len(train_files)}")
print(f"测试集样本数量: {len(val_files)}")

# 将训练集文件路径写入train.txt文件
with open(train_txt_path, 'w') as train_file:
    for file_path, category in train_files:
        train_file.write(f"{file_path} {category}\n")

# 将测试集文件路径写入val.txt文件
with open(val_txt_path, 'w') as val_file:
    for file_path, category in val_files:
        val_file.write(f"{file_path} {category}\n")

print("文件路径已保存至 train.txt 和 val.txt")
