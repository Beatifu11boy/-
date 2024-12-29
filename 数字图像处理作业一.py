import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import cv2
import math
from PIL import Image

# def img_transform(img,lower_bound,upper_bound):
#     i_img=np.zeros_like(img,dtype=np.uint8)
#     mask=(img>=lower_bound)&(img<=upper_bound)
#     i_img[mask]=255-img[mask]
#     i_img[~mask]=img[~mask]
#     return i_img

img=cv.imread('D01961BF87B3C67FF7C4D11DF709ACCE.jpg',cv.IMREAD_GRAYSCALE)
# new_img=img_transform(img,0,255)

# cv.imshow('Original',img)
# cv.imshow('trans',new_img)
# cv.waitKey(0)
# cv.destroyAllWindows()


# gamma_value = [ 0.5,1.5]
# normImg = lambda x: 255. * (x-x.min()) / (x.max()-x.min()+1e-6)
# plt.figure(figsize=(9,6))
# for k in range(len(gamma_value)):
#     imgGamma = np.power(img, gamma_value[k])
#     imgGamma = np.uint8(normImg(imgGamma))
#     plt.subplot(2, 3, k+1), plt.axis('off')
#     plt.imshow(imgGamma,  cmap='gray', vmin=0, vmax=255)
#     plt.title(f"gamma={gamma_value[k]}")
# plt.show()

# r, c = img.shape  # 图像预处理
# x = np.zeros((r, c, 8), dtype=np.uint8)  # 构造提取矩阵
# for i in range(8):
#     x[:, :, i] = 2 ** i
# # print(x.shape)
#
# w = np.zeros((r, c, 8), dtype=np.uint8)
# # print(r)
# plt.figure(figsize=(9,6))
# for i in range(8):
#     w[:, :, i] = cv.bitwise_and(img, x[:, :, i])  # 提取位平面
#     mask = w[:, :, i] > 0  # 阈值处理
#     w[mask] = 255
#     cv.imshow(str(i), w[:, :, i])  # 显示图片
# cv.waitKey(0)
# cv.destroyAllWindows()


# 导入原始图像,色彩空间为灰度图
# 调用cv2.calcHist 函数绘制直方图
# img_hist = cv.calcHist([img], [0], None, [256], [0, 256])
#
# # 直方图均衡化,调用cv2.equalizeHist 函数实心
# result_img = cv.equalizeHist(img)
# # 显示原始图像
# cv.imshow('img', img)
# # 显示均衡化后的图像
# cv.imshow('result_img', result_img)
# cv.waitKey(0)
# # 用蓝色绘制原始图像直方图
# plt.plot(img_hist, color="b")
# plt.title('1')
# plt.show()
#
# plt.hist(img_hist.ravel(), 256, [0, 256])
# plt.title('2')
# plt.show()
# # 绘制均衡化后的直方图
#
# plt.hist(result_img.ravel(), 256, [0, 256])
# plt.title('3')
# plt.show()



# dst1 = cv.blur(img, (3, 3))  # 使用大小为3*3的滤波核进行均值滤波
# dst2 = cv.blur(img, (5, 5))  # 使用大小为5*5的滤波核进行均值滤波
# dst3 = cv.blur(img, (9, 9))  # 使用大小为9*9的滤波核进行均值滤波
# cv.imshow("img", img)
# cv.imshow("3*3", dst1)  # 显示滤波效果
# cv.imshow("5*5", dst2)
# cv.imshow("9*9", dst3)
# cv.waitKey()
# cv.destroyAllWindows()


#
# dst1 = cv.GaussianBlur(img, (5, 5), 0, 0)  # 使用大小为5*5的滤波核进行高斯滤波
# dst2 = cv.GaussianBlur(img, (9, 9), 0, 0)  # 使用大小为9*9的滤波核进行高斯滤波
# dst3 = cv.GaussianBlur(img, (15, 15), 0, 0)  # 使用大小为15*15的滤波核进行高斯滤波
# cv.imshow("img", img)
# cv.imshow("5", dst1)
# cv.imshow("9", dst2)
# cv.imshow("15", dst3)
# cv.waitKey()
# cv.destroyAllWindows()
#
#
#
#
# dst1 = cv.boxFilter(img, -1,(3, 3))  # 使用大小为3*3的滤波核进行均值滤波
# dst2 = cv.boxFilter(img,-1, (11, 11))  # 使用大小为5*5的滤波核进行均值滤波
# dst3 = cv.boxFilter(img, -1,(121, 121))  # 使用大小为9*9的滤波核进行均值滤波
# cv.imshow("img", img)
# cv.imshow("3*3", dst1)  # 显示滤波效果
# cv.imshow("5*5", dst2)
# cv.imshow("9*9", dst3)
# cv.waitKey()
# cv.destroyAllWindows()

# roberts
# 定义Roberts算子的卷积核
# roberts_x = np.array([[0, 1], [-1, 0]], dtype=np.float32)
# roberts_y = np.array([[1, 0], [ 0, -1]], dtype=np.float32)
# # 函数cv2.filter2D()来实现卷积操作，可以使用该函数来实现Roberts交叉算子
# roberts_img_x = cv.filter2D(img, -1, roberts_x)
# roberts_img_y = cv.filter2D(img, -1, roberts_y)
# # 将图像的像素值缩放到指定范围内
# roberts_img = cv.addWeighted(roberts_img_x, 0.5, roberts_img_y, 0.5, 0)
# # 显示Roberts交叉算子的梯度图
# cv.imshow('Roberts Gradient', roberts_img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# # prewitt
# prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
# prewitt_y = np.array([[ 1, 1, 1], [ 0, 0, 0], [-1, -1, -1]], dtype=np.float32)
# # 函数cv2.filter2D()来实现卷积操作，可以使用该函数来实现Prewitt算子
# prewitt_img_x = cv.filter2D(img, -1, prewitt_x)
# prewitt_img_y = cv.filter2D(img, -1, prewitt_y)
# prewitt_img = cv.addWeighted(prewitt_img_x, 0.5, prewitt_img_y, 0.5, 0)
# # 显示Prewitt算子的梯度图
# cv.imshow('Prewitt Gradient', prewitt_img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# # # sobel
# sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
# # 使用Sobel算子在y方向上求导
# sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
# # 计算每个像素的梯度大小
# sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
# # 将梯度大小映射到0-255之间，并转换为8位无符号整型
# sobel = cv.normalize(sobel, None, 0, 255, cv.NORM_MINMAX)
# sobel = np.uint8(sobel)
# # 显示结果
# cv.imshow('Sobel', sobel)
# cv.waitKey(0)
# cv.destroyAllWindows()
# #

# def krisch_edge_detection(image):
#
#     ks=[
#     [[-3, -3, -3], [5, 0, -3], [5, 5, -3]],
#     [[5, -3, -3], [5, 0, -3], [5, -3, -3]],
#     [[5, 5, -3], [5, 0, -3], [-3, -3, -3]],
#     [[5  ,5 ,5], [-3, 0, -3], [-3,-3, -3]],
#     [[-3, 5 ,5], [-3 ,0, 5], [-3, -3, -3]],
#     [[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]],
#     [[-3, -3, -3], [-3, 0, 5], [-3 ,5, 5]],
#     [[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]]
#
#     result=image.copy()
#     for k in ks:
#         k=np.array(k)
#         out=cv.filter2D(image,-1,k)
#         result= np.maximum(result,out)
#     return result
#
# # 读取图像
#
# edge_image = krisch_edge_detection(img)
#
# # 显示结果
# cv.imshow("Original", img)
# cv.imshow("krisch", edge_image)
# cv.waitKey(0)
# cv.destroyAllWindows()

# 拉普拉斯
# def laplace(image):
#     k=np.array([[0,1,0],[1,-4,1],[0,1,0]])
#     # k = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
#     out = cv.filter2D(image, -1, k)
#     return out
#
# delt_f=laplace(img)
# g_img = cv.addWeighted(delt_f, -1, img, 1, 0)
# cv.imshow("Original", img)
# # cv.imshow("拉普拉斯变换后的图", delt_f)
# cv.imshow("锐化后的图像", g_img)
# cv.imwrite('image/kjyu_laplace.png',g_img)
# cv.waitKey(0)
# cv.destroyAllWindows()


image = cv2.imread('D01961BF87B3C67FF7C4D11DF709ACCE.jpg',cv2.IMREAD_GRAYSCALE)
#
# # 读取图片
# image = cv2.imread('D01961BF87B3C67FF7C4D11DF709ACCE.jpg')  # 请将 'image.jpg' 替换为您的图片路径
#
# # 分离 RGB 通道
# b, g, r = cv2.split(image)
# # 显示 RGB 分量图
# plt.subplot(2, 2, 1)
# plt.imshow(r, cmap='gray')
# plt.title('R')
#
# plt.subplot(2, 2, 2)
# plt.imshow(g, cmap='gray')
# plt.title('G')
#
# plt.subplot(2, 2, 3)
# plt.imshow(b, cmap='gray')
# plt.title('B')
#
# plt.subplot(2, 2, 4)
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title('Original Image')
#
# plt.show()
# #
#
# def rgb2hsi(rgb_image):
#     # 将 RGB 图像转换到 HSI 颜色空间
#     r, g, b = cv2.split(rgb_image)
#     r_ = r / 255.
#     g_ = g / 255.
#     b_ = b / 255.
#
#     num = 0.5 * ((r_ - g_) + (r_ - b_))
#     den = np.sqrt((r_ - g_)**2 + (r_ - b_) * (g_ - b_))
#     theta = np.arccos(num / (den + 1e-10))
#
#     h = np.zeros_like(r)
#     h[b_ > g_] = 2 * np.pi - theta[b_ > g_]
#     h[b_ <= g_] = theta[b_ <= g_]
#     h = h / (2 * np.pi)
#
#     i = (r_ + g_ + b_) / 3
#
#     s = 1 - np.minimum(np.minimum(r_, g_), b_) / (i + 1e-10)
#     return h, s, i
#
#
# def hsi2rgb(H, S, I):
#     R = G = B = np.zeros_like(I)
#     for i in range(H.shape[0]):
#         for j in range(H.shape[1]):
#             if S[i, j] == 0:
#                 R[i, j] = G[i, j] = B[i, j] = I[i, j]  # 灰度图
#             else:
#                 H_deg = H[i, j] * (180 / np.pi)  # H转换为度数
#                 if H_deg < 120:
#                     B[i, j] = I[i, j] * (1 - S[i, j])
#                     R[i, j] = I[i, j] * (
#                                 1 + (S[i, j] * np.cos(H_deg * np.pi / 180)) / np.cos((60 - H_deg) * np.pi / 180))
#                     G[i, j] = 3 * I[i, j] - (R[i, j] + B[i, j])
#                 elif H_deg < 240:
#                     H_deg -= 120
#                     R[i, j] = I[i, j] * (1 - S[i, j])
#                     G[i, j] = I[i, j] * (
#                                 1 + (S[i, j] * np.cos(H_deg * np.pi / 180)) / np.cos((60 - H_deg) * np.pi / 180))
#                     B[i, j] = 3 * I[i, j] - (R[i, j] + G[i, j])
#                 else:
#                     H_deg -= 240
#                     G[i, j] = I[i, j] * (1 - S[i, j])
#                     B[i, j] = I[i, j] * (
#                                 1 + (S[i, j] * np.cos(H_deg * np.pi / 180)) / np.cos((60 - H_deg) * np.pi / 180))
#                     R[i, j] = 3 * I[i, j] - (G[i, j] + B[i, j])
#
#     return R,G,B
#
# h, s, i = rgb2hsi(image)
#
# # 显示 HSI 分量图
# plt.subplot(2, 2, 1)
# plt.imshow(h, cmap='gray')
# plt.title('H')
#
# plt.subplot(2, 2, 2)
# plt.imshow(s, cmap='gray')
# plt.title('S')
#
# plt.subplot(2, 2, 3)
# plt.imshow(i, cmap='gray')
# plt.title('I')
#
# plt.subplot(2, 2, 4)
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title('Original Image')
#
# plt.show()
#
# def histogram_equalization_rgb(image_path):
#     # 读取图像
#     image = cv2.imread(image_path)
#     # 分别对三个通道进行直方图均衡化
#     r_channel, g_channel, b_channel = cv2.split(image)
#     r_channel_equalized = cv2.equalizeHist(r_channel)
#     g_channel_equalized = cv2.equalizeHist(g_channel)
#     b_channel_equalized = cv2.equalizeHist(b_channel)
#     # 合并均衡化后的通道
#     equalized_image = cv2.merge((r_channel_equalized, g_channel_equalized, b_channel_equalized))
#     # 显示原始图像和均衡化后的图像
#     cv2.imshow("Original Image", image)
#     cv2.imshow("RGB Image", equalized_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# histogram_equalization_rgb('D01961BF87B3C67FF7C4D11DF709ACCE.jpg')
#
# def hsi_histogram_equalization(image_path):
#     # 读取图像
#     image = cv2.imread(image_path)
#     # 将图像从 BGR 转换为 HSV 色彩空间
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     # 提取亮度通道（V 通道）
#     v_channel = hsv_image[:, :, 2]
#     # 对亮度通道进行直方图均衡化
#     equalized_v_channel = cv2.equalizeHist(v_channel)
#     # 将均衡化后的亮度通道替换原来的 V 通道
#     hsv_image[:, :, 2] = equalized_v_channel
#     # 将处理后的 HSV 图像转换回 BGR 色彩空间
#     equalized_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
#     # 显示原始图像和均衡化后的图像
#     cv2.imshow("Original Image", image)
#     cv2.imshow("HSI Image", equalized_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# hsi_histogram_equalization('D01961BF87B3C67FF7C4D11DF709ACCE.jpg')



# 均值滤波器
# 定义滤波器窗口大小
# kernel_size = (5, 5)
#
# # 使用均值滤波
# #  使用 cv2.boxFilter（可以提供更细致的控制）
# mean_filtered_image_alt = cv2.boxFilter(image, ddepth=-1, ksize=kernel_size, normalize=True)
#
# # 显示原始图像和均值滤波后的图像
# cv2.imshow('Original Image', image)
# cv2.imshow('Mean Filtered Image', mean_filtered_image_alt)
#
# # 等待按键并关闭所有窗口
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # 拉普拉斯变换
#
# # 分离 RGB 通道
# b_channel, g_channel, r_channel = cv2.split(image)
#
# # 分别对每个通道应用拉普拉斯变换
# laplacian_b = cv2.Laplacian(b_channel, cv2.CV_64F)
# laplacian_g = cv2.Laplacian(g_channel, cv2.CV_64F)
# laplacian_r = cv2.Laplacian(r_channel, cv2.CV_64F)
#
# # 转换为 uint8 类型
# laplacian_b = cv2.convertScaleAbs(laplacian_b)
# laplacian_g = cv2.convertScaleAbs(laplacian_g)
# laplacian_r = cv2.convertScaleAbs(laplacian_r)
#
# # 合并通道
# laplacian_image_rgb = cv2.merge((laplacian_b, laplacian_g, laplacian_r))
# a=cv2.addWeighted(image,1,laplacian_image_rgb,0.5,0)
# # 显示合并后的图像
# cv2.imshow('or',image)
# cv2.imshow('ts',a)
#
# # cv2.imshow('Laplacian Image (RGB)', laplacian_image_rgb)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
#
# def rgb_to_hsi(image):
#     image = image.astype('float') / 255.0
#     R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
#     I = (R + G + B) / 3.0
#
#     # 计算色相和饱和度
#     num = 0.5 * ((R - G) + (R - B))
#     den = np.sqrt((R - G) ** 2 + (R - B) * (G - B))
#     theta = np.arccos(num / (den + 1e-6))  # 阻止除零
#     H = np.zeros_like(I)
#
#     H[B <= G] = theta[B <= G]
#     H[B > G] = 2 * np.pi - theta[B > G]
#
#     S = 1 - (3 / (R + G + B + 1e-6)) * np.minimum(np.minimum(R, G), B)
#
#     return H, S, I
#
#
# def hsi_to_rgb(H, S, I):
#     R = np.zeros_like(I)
#     G = np.zeros_like(I)
#     B = np.zeros_like(I)
#
#     for i in range(H.shape[0]):
#         for j in range(H.shape[1]):
#             if S[i, j] == 0:  # 灰度图
#                 R[i, j] = I[i, j]
#                 G[i, j] = I[i, j]
#                 B[i, j] = I[i, j]
#             else:
#                 H_deg = H[i, j] * (180 / np.pi)  # 弧度转度
#                 if H_deg < 120:
#                     B[i, j] = I[i, j] * (1 - S[i, j])
#                     R[i, j] = I[i, j] * (
#                                 1 + (S[i, j] * np.cos(H_deg * np.pi / 180)) / np.cos((60 - H_deg) * np.pi / 180))
#                     G[i, j] = 3 * I[i, j] - (R[i, j] + B[i, j])
#                 elif H_deg < 240:
#                     H_deg -= 120
#                     R[i, j] = I[i, j] * (1 - S[i, j])
#                     G[i, j] = I[i, j] * (
#                                 1 + (S[i, j] * np.cos(H_deg * np.pi / 180)) / np.cos((60 - H_deg) * np.pi / 180))
#                     B[i, j] = 3 * I[i, j] - (R[i, j] + G[i, j])
#                 else:
#                     H_deg -= 240
#                     G[i, j] = I[i, j] * (1 - S[i, j])
#                     B[i, j] = I[i, j] * (
#                                 1 + (S[i, j] * np.cos(H_deg * np.pi / 180)) / np.cos((60 - H_deg) * np.pi / 180))
#                     R[i, j] = 3 * I[i, j] - (G[i, j] + B[i, j])
#
#     return np.clip(np.stack((R, G, B), axis=-1) * 255, 0, 255).astype(np.uint8)
#
#
#
# # 转换为HSI
# H, S, I = rgb_to_hsi(image)
#
# # 应用均值滤波到强度分量I
# kernel_size = (5, 5)  # 设置卷积核大小
# I_blurred = cv2.blur(I, kernel_size)
#
# # 重建HSI图像，保持H和S不变
# H_new, S_new, I_new = H, S, I_blurred
#
# # 转换回RGB颜色空间
# output_image = hsi_to_rgb(H_new, S_new, I_new)
#
# # 显示原图和处理后的图像
# cv2.imshow('Original Image', image)
# cv2.imshow('HSI JUNZHI Image', output_image)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# # 应用拉普拉斯变换到强度分量I
# I_laplacian = cv2.Laplacian(I, cv2.CV_64F)
#
# # 归一化拉普拉斯变换结果到范围[0, 1]
# I_laplacian_normalized = cv2.normalize(I_laplacian, None, 0, 1, cv2.NORM_MINMAX)
#
# # 重建HSI图像，保持H和S不变
# H_new, S_new, I_new = H, S, I_laplacian_normalized
#
# # 转换回RGB颜色空间
# output_image = hsi_to_rgb(H_new, S_new, I_new)
#
# # 显示原图和处理后的图像
# cv2.imshow('Original Image', image)
# cv2.imshow('HSI with Laplacian', output_image)
#
# # 等待键盘事件并关闭窗口
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# dft

# def dft(image):
#     f = np.fft.fft2(image)
#     fshift = np.fft.fftshift(f)
#     magnitude_spectrum = 20*np.log(np.abs(fshift))
#     return magnitude_spectrum
#
# magnitude_spectrum = dft(image)
#
# plt.subplot(121), plt.imshow(image, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
#
# plt.show()
#
#
# # dft 逆
# dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
# dft_shift = np.fft.fftshift(dft)
#
# idft_shift = np.fft.ifftshift(dft_shift)
# idft = cv2.idft(idft_shift)
# recovered_img = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])
# # 显示原始图像、逆变换恢复的图像
# plt.figure(figsize=(10, 5))
# plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
# plt.subplot(122), plt.imshow(recovered_img, cmap='gray'), plt.title('F-1')
# plt.show()

#      理想滤波器

# def ideal_low_pass_filter(image, cutoff_frequency):
#     # 进行傅里叶变换
#     f = np.fft.fft2(image)
#     fshift = np.fft.fftshift(f)
#
#     rows, cols = image.shape
#     crow, ccol = rows // 2, cols // 2  # 中心位置
#
#     # 创建滤波器
#     mask = np.zeros((rows, cols), np.uint8)
#     for i in range(rows):
#         for j in range(cols):
#             distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
#             if distance <= cutoff_frequency:
#                 mask[i, j] = 1
#
#     # 应用滤波器
#     fshift = fshift * mask
#     f_ishift = np.fft.ifftshift(fshift)
#     img_back = np.fft.ifft2(f_ishift)
#     img_back = np.abs(img_back)
#     return img_back
#
# cutoff_frequencies = [10, 60,160]  # 您可以自定义不同的截止频率
# fig, axs = plt.subplots(1, len(cutoff_frequencies) + 1, figsize=(15, 5))
# # 显示原始图像
# axs[0].imshow(image, cmap='gray')
# axs[0].set_title('Original Image')
# axs[0].axis('off')
# for i, cutoff_frequency in enumerate(cutoff_frequencies):
#     filtered_image = ideal_low_pass_filter(image, cutoff_frequency)
#     axs[i + 1].imshow(filtered_image, cmap='gray')
#     axs[i + 1].set_title(f'ILPF (Cutoff: {cutoff_frequency})')
#     axs[i + 1].axis('off')
# plt.show()
#
#
# #                         BLPF
#
# def BLPD(image,cutoff_frequency,filter_order_list):
#     # 傅里叶变换图像
#     image_frequency = np.fft.fftshift(np.fft.fft2(image))
#     # 计算图像的中心坐标
#     center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
#     for filter_order in filter_order_list:
#         # 创建巴特沃斯低通滤波器
#         butterworth_filter = np.zeros_like(image, dtype=np.float32)
#         for y in range(image.shape[0]):
#             for x in range(image.shape[1]):
#                 distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
#                 # 计算巴特沃斯滤波器的值
#                 butterworth_filter[y, x] = 1 / (1 + (distance / cutoff_frequency) ** (2 * filter_order))
#         # 应用滤波器
#         filtered_image_frequency = image_frequency * butterworth_filter
#         # 反傅里叶变换
#         filtered_image = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_image_frequency)))
#         return filtered_image
#
# cutoff_frequencies = [10,60,160]  # 设置取样的截止频率
# filter_order_list = [10]  # 设置取样的阶数5
#
#
# fig, axs = plt.subplots(1, len(cutoff_frequencies) + 1, figsize=(15, 5))
# # 显示原始图像
# axs[0].imshow(image, cmap='gray')
# axs[0].set_title('Original Image')
# axs[0].axis('off')
# for i, cutoff_frequency in enumerate(cutoff_frequencies):
#     filtered_image = BLPD(image, cutoff_frequency,filter_order_list)
#     axs[i + 1].imshow(filtered_image, cmap='gray')
#     axs[i + 1].set_title(f'BLPF10 (Cutoff: {cutoff_frequency})')
#     axs[i + 1].axis('off')
# plt.show()
#
#                                        # GLPF
#
# def gaussian_lowpass_filter(image, cutoff_frequency):
#     # 进行傅里叶变换
#     f = np.fft.fft2(image)
#     fshift = np.fft.fftshift(f)
#
#     rows, cols = image.shape
#     crow, ccol = rows // 2, cols // 2  # 中心位置
#
#     # 构建高斯滤波器
#     gaussian_filter = np.zeros((rows, cols))
#     for i in range(rows):
#         for j in range(cols):
#             d = np.sqrt((i - crow)**2 + (j - ccol)**2)
#             gaussian_filter[i, j] = np.exp(-(d**2) / (2 * (cutoff_frequency**2)))
#
#     # 应用滤波器
#     fshift_filtered = fshift * gaussian_filter
#
#     # 逆傅里叶变换
#     f_inverse_shift = np.fft.ifftshift(fshift_filtered)
#     filtered_image = np.fft.ifft2(f_inverse_shift)
#     filtered_image = np.abs(filtered_image)  # 取绝对值
#
#     return filtered_image
#
# cutoff_frequencies = [10,60,150]  # 截止频率，可根据需要调整
# fig, axs = plt.subplots(1, len(cutoff_frequencies) + 1, figsize=(15, 5))
# # 显示原始图像
# axs[0].imshow(image, cmap='gray')
# axs[0].set_title('Original Image')
# axs[0].axis('off')
# for i, cutoff_frequency in enumerate(cutoff_frequencies):
#     filtered_image = gaussian_lowpass_filter(image, cutoff_frequency)
#     axs[i + 1].imshow(filtered_image, cmap='gray')
#     axs[i + 1].set_title(f'GLPF (Cutoff: {cutoff_frequency})')
#     axs[i + 1].axis('off')
# plt.show()








# 傅里叶变换得到频谱

img_f = np.fft.fftshift(np.fft.fft2(image.astype(np.float32)))
# image_frequency = np.fft.fftshift(np.fft.fft2(image))
# 获取图像大小
m, n = img_f.shape
O_x, O_y = m // 2, n // 2  # 获取圆心坐标
a = np.max(np.abs(img_f))  # 求img_f得最大值

# 提前定义滤波后的频谱
img = np.zeros_like(img_f)

# 计算拉普拉斯滤波器并应用
for j in range(n):
    for i in range(m):
        d = np.sqrt((i - O_x)**2 + (j - O_y)**2)  # 计算两点之间的距离
        H_ij = -4 * np.pi**2 * d**2 / a  # 拉普拉斯滤波器
        img[i, j] = (1 - H_ij) * img_f[i, j]

# 傅里叶反变换
img = np.fft.ifftshift(img)
img = np.abs(np.fft.ifft2(img))
# filtered_image = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_image_frequency)))
# 将结果转换为uint8类型以便显示
img = img.astype(np.uint8)

# 显示原图和处理后的图像
cv2.imshow('Original ', image)
cv2.imshow('Laplacian ts', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


image1 = cv2.imread('image/kjyu_laplace.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('image/laplace_ts.png', cv2.IMREAD_GRAYSCALE)

# 计算图像差异
diff = cv2.absdiff(image1, image2)

# 显示差异图像
cv2.imshow('Difference', diff)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 可选：查找差异图像的轮廓
_, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(diff, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 再次显示带有轮廓的差异图像
cv2.imshow('Difference with Contours', diff)
cv2.waitKey(0)
cv2.destroyAllWindows()

















