import cv2
import numpy as np

# 读取灰度图像
image = cv2.imread('/media/data2/lcl_e/gl/BUSSAM/outputs/337_25.png', cv2.IMREAD_GRAYSCALE)

# 将像素值大于128的设为1，其余设为0
binary_image = np.where(image > 128, 1, 0).astype(np.uint8)

# 保存为二值图
cv2.imwrite('/media/data2/lcl_e/gl/BUSSAM/outputs/binary_image_337_25.png', binary_image * 255)

print("Binary image saved successfully.")