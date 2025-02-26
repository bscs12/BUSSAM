# 本文件用于为数据集生成txt文件


import os

# 读取一个文件夹中的所有文件
input_folder_path = 'input_folder'  # 替换为你的输入文件夹路径
output_txt_path = 'output.txt'  # 替换为你的输出txt文件路径

try:
    # 遍历输入文件夹中的所有文件
    file_names = []
    for filename in os.listdir(input_folder_path):
        # 获取每个文件的文件名（不包括扩展名）
        file_name, file_extension = os.path.splitext(filename)
        file_names.append(file_name)

    # 文件名按照从小到大的顺序排列
    sorted_file_names = sorted(file_names)

    # 将每个文件的文件名保存到一个txt文件，每个文件名占一行
    with open(output_txt_path, 'w', encoding='utf-8') as output_file:
        output_file.write('\n'.join(sorted_file_names))

    print(f"文件名已保存到: {output_txt_path}")

except FileNotFoundError:
    print("找不到输入文件夹:", input_folder_path)
except Exception as e:
    print("发生错误:", str(e))
