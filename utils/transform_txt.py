# 本文件用于txt文件添加前缀字符串或删除字符串


# 读取txt文件
input_file_path = '/media/data2/lcl_e/gl/BUSSAM/datasets/MainPatient/BUSI_test.txt'  # 替换为你的输入文件路径
output_file_path = '/media/data2/lcl_e/gl/BUSSAM/test.txt'  # 替换为你的输出文件路径

try:
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        lines = input_file.readlines()

    # 在每一行开头添加前缀字符串'prefix'
    # modified_lines = ['prefix' + line for line in lines]

    # 删除字符串'string_to_remove'
    modified_lines = [line.replace('BUSI/', '') for line in lines]

    # 将修改后的文件保存到另一个目录
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.writelines(modified_lines)

    print("文件处理完成，已保存到", output_file_path)

except FileNotFoundError:
    print("找不到输入文件:", input_file_path)
except Exception as e:
    print("发生错误:", str(e))
