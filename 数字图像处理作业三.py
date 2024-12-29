import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import cv2
import math
from PIL import Image

color_image=cv2.imread('D01961BF87B3C67FF7C4D11DF709ACCE.jpg')
image=cv.imread('D01961BF87B3C67FF7C4D11DF709ACCE.jpg',cv.IMREAD_GRAYSCALE)

#                                 prewitt

# prewitt_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
# prewitt_x = np.array([[ 1, 1, 1], [ 0, 0, 0], [-1, -1, -1]], dtype=np.float32)
# # 函数cv2.filter2D()来实现卷积操作，可以使用该函数来实现Prewitt算子
# prewitt_img_x = cv.filter2D(img, -1, prewitt_x)
# prewitt_img_y = cv.filter2D(img, -1, prewitt_y)
# prewitt_img = cv.addWeighted(prewitt_img_x, 1, prewitt_img_y, 1, 0)
# 显示Prewitt算子的梯度图
# cv.imshow('x', prewitt_img_x)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# cv.imshow('y', prewitt_img_y)
# cv.waitKey(0)
# cv.destroyAllWindows()


# cv.imshow('Prewitt Gradient', prewitt_img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
#
# threshold_value = 120
# _, binary_image = cv2.threshold(prewitt_img, threshold_value, 255, cv2.THRESH_BINARY)
# cv2.imshow('Binary Image', binary_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#                        prewitt斜向处理
# prewitt_45 = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]], dtype=np.float32)
# prewitt_135 = np.array([[ 0, 1, 1], [ -1, 0, 1], [-1, -1, 0]], dtype=np.float32)
# # 函数cv2.filter2D()来实现卷积操作，可以使用该函数来实现Prewitt算子
# prewitt_img_45 = cv.filter2D(img, -1, prewitt_45)
# prewitt_img_135 = cv.filter2D(img, -1, prewitt_135)
# prewitt_img_xie = cv.addWeighted(prewitt_img_45, 0.5, prewitt_img_135, 0.5, 0)
# # 显示Prewitt算子的梯度图
# cv.imshow('45', prewitt_img_45)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# cv.imshow('135', prewitt_img_135)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# cv.imshow('Prewitt Gradient', prewitt_img_xie)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# threshold_value = 80
# _, binary_image = cv2.threshold(prewitt_img_xie, threshold_value, 255, cv2.THRESH_BINARY)
# cv2.imshow('Binary Image', binary_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

                                                    # canny
# 高斯滤波
# image_blur = cv2.GaussianBlur(image, (5, 5), 0)
# # 计算梯度
# gradient_x = cv2.Sobel(image_blur, cv2.CV_64F, 1, 0, ksize=3)
# gradient_y = cv2.Sobel(image_blur, cv2.CV_64F, 0, 1, ksize=3)
# gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
# gradient_direction = np.arctan2(gradient_y, gradient_x) * (180/np.pi)
# # 非极大值抑制
# gradient_magnitude_suppressed = np.copy(gradient_magnitude)
# for i in range(1, gradient_magnitude.shape[0]-1):
#     for j in range(1, gradient_magnitude.shape[1]-1):
#         direction = gradient_direction[i, j]
#         if (0 <= direction < 22.5) or (157.5 <= direction <= 180) or (-22.5 <= direction < 0) or (-180 <= direction < -157.5):
#             if (gradient_magnitude[i, j] < gradient_magnitude[i, j-1]) or (gradient_magnitude[i, j] < gradient_magnitude[i, j+1]):
#                 gradient_magnitude_suppressed[i, j] = 0
#         elif (22.5 <= direction < 67.5) or (-157.5 <= direction < -112.5):
#             if (gradient_magnitude[i, j] < gradient_magnitude[i-1, j+1]) or (gradient_magnitude[i, j] < gradient_magnitude[i+1, j-1]):
#                 gradient_magnitude_suppressed[i, j] = 0
#         elif (67.5 <= direction < 112.5) or (-112.5 <= direction < -67.5):
#             if (gradient_magnitude[i, j] < gradient_magnitude[i-1, j]) or (gradient_magnitude[i, j] < gradient_magnitude[i+1, j]):
#                 gradient_magnitude_suppressed[i, j] = 0
#         elif (112.5 <= direction < 157.5) or (-67.5 <= direction < -22.5):
#             if (gradient_magnitude[i, j] < gradient_magnitude[i-1, j-1]) or (gradient_magnitude[i, j] < gradient_magnitude[i+1, j+1]):
#                 gradient_magnitude_suppressed[i, j] = 0
# # 双阈值检测
# low_threshold = 100
# high_threshold = 200
# canny_edges = np.zeros_like(gradient_magnitude_suppressed)
# canny_edges[(gradient_magnitude_suppressed >= high_threshold)] = 255
# canny_edges[(gradient_magnitude_suppressed >= low_threshold) & (gradient_magnitude_suppressed < high_threshold)] = 127
# # 边缘连接
# def edge_linking(i, j):
#     if canny_edges[i, j] == 127:
#         canny_edges[i, j] = 255
#         for x in range(i-1, i+2):
#             for y in range(j-1, j+2):
#                 if canny_edges[x, y] == 127:
#                     edge_linking(x, y)
# for i in range(1, canny_edges.shape[0]-1):
#     for j in range(1, canny_edges.shape[1]-1):
#         if canny_edges[i, j] == 255:
#             edge_linking(i, j)
# # 显示结果
# cv2.imshow('Original Image', image)
# cv2.imshow('Canny Edges', canny_edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# canny_edges = cv2.Canny(image, 100, 200)
# cv2.imshow('canny',canny_edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# rows, cols = image.shape[:2]
# gray_hist = np.zeros([256], np.uint64)
# for i in range(rows):
#     for j in range(cols):
#         gray_hist[image[i][j]] += 1
# uniformGrayHist = gray_hist / float(rows * cols)
# # 计算零阶累积距和一阶累积距
# zeroCumuMomnet = np.zeros(256, np.float32)
# oneCumuMomnet = np.zeros(256, np.float32)
# for k in range(256):
#     if k == 0:
#         zeroCumuMomnet[k] = uniformGrayHist[0]
#         oneCumuMomnet[k] = (k) * uniformGrayHist[0]
#     else:
#         zeroCumuMomnet[k] = zeroCumuMomnet[k - 1] + uniformGrayHist[k]
#         oneCumuMomnet[k] = oneCumuMomnet[k - 1] + k * uniformGrayHist[k]
# # 计算类间方差
# variance = np.zeros(256, np.float32)
# for k in range(255):
#     if zeroCumuMomnet[k] == 0 or zeroCumuMomnet[k] == 1:
#         variance[k] = 0
#     else:
#         variance[k] = math.pow(oneCumuMomnet[255] * zeroCumuMomnet[k] - oneCumuMomnet[k], 2) / (
#                     zeroCumuMomnet[k] * (1.0 - zeroCumuMomnet[k]))
# # 找到阈值
# threshLoc = np.where(variance[0:255] == np.max(variance[0:255]))
# thresh = threshLoc[0][0]
# # 阈值处理
# threshold = np.copy(image)
# print(thresh)
# threshold[threshold > thresh] = 255
# threshold[threshold <= thresh] = 0
# cv2.imshow("test", threshold)
# cv2.waitKey(0)


# 计算梯度
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# 计算梯度幅度
gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
gradient_magnitude = np.uint8(gradient_magnitude)
cv.imshow('梯度图', gradient_magnitude)
cv.waitKey(0)
cv.destroyAllWindows()

# 手动指定阈值
T = 100  # 这个值可以根据具体情况调整

# 利用指定的阈值进行二值化
_, binary_image = cv2.threshold(gradient_magnitude, T, 255, cv2.THRESH_BINARY)
cv.imshow('二值图', binary_image)
cv.waitKey(0)
cv.destroyAllWindows()

# 计算直方图，仅考虑二值图像中为255的位置
histogram = cv2.calcHist([image], [0], binary_image, [256], [0, 256])
# 显示直方图
plt.figure(figsize=(10, 5))
plt.title("Histogram of Strong Edge Pixels")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.plot(histogram)
plt.xlim([0, 256])  # 限制 x 轴范围
plt.show()


output_image = np.zeros_like(image)
output_image[binary_image == 255] = image[binary_image == 255]
cv2.imshow('mask',output_image)
cv.waitKey(0)
cv.destroyAllWindows()






new=cv2.calcHist([output_image],[0],None, [256], [0, 256])
plt.figure(figsize=(10, 5))
plt.title("Histogram of Strong Edge Pixels")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.plot(new)
plt.xlim([0, 2])  # 限制 x 轴范围
plt.show()

rows, cols = output_image.shape[:2]
gray_hist = np.zeros([256], np.uint64)
for i in range(rows):
    for j in range(cols):
        gray_hist[output_image[i][j]] += 1
uniformGrayHist = gray_hist / float(rows * cols)
# 计算零阶累积距和一阶累积距
zeroCumuMomnet = np.zeros(256, np.float32)
oneCumuMomnet = np.zeros(256, np.float32)
for k in range(256):
    if k == 0:
        zeroCumuMomnet[k] = uniformGrayHist[0]
        oneCumuMomnet[k] = (k) * uniformGrayHist[0]
    else:
        zeroCumuMomnet[k] = zeroCumuMomnet[k - 1] + uniformGrayHist[k]
        oneCumuMomnet[k] = oneCumuMomnet[k - 1] + k * uniformGrayHist[k]
# 计算类间方差
variance = np.zeros(256, np.float32)
for k in range(255):
    if zeroCumuMomnet[k] == 0 or zeroCumuMomnet[k] == 1:
        variance[k] = 0
    else:
        variance[k] = math.pow(oneCumuMomnet[255] * zeroCumuMomnet[k] - oneCumuMomnet[k], 2) / (
                    zeroCumuMomnet[k] * (1.0 - zeroCumuMomnet[k]))
# 找到阈值
threshLoc = np.where(variance[0:255] == np.max(variance[0:255]))
thresh = threshLoc[0][0]
# 阈值处理
threshold = np.copy(image)
print(thresh)
threshold[threshold > thresh] = 255
threshold[threshold <= thresh] = 0
cv2.imshow("test", threshold)
cv2.waitKey(0)










# 使用 Otsu 方法进行全局阈值分割
# otsu_threshold, otsu_binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)
# print(otsu_threshold)
# cv.imshow('otsu——image', otsu_binary_image)
# cv.waitKey(0)
# cv.destroyAllWindows()

