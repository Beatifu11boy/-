# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
#
# # 1. 读取彩色图像并转换为灰度图像
# img = cv2.imread('D01961BF87B3C67FF7C4D11DF709ACCE.jpg')  # 请替换为实际图像路径
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # 2. 获取图像的尺寸
# height, width = gray_img.shape
#
# # 3. 将图像展平成二维数据 (height * width, 1)
# data = gray_img.reshape(height, width)  # 此时数据是 (height, width)
#
# # 4. 数据中心化：计算每列的均值并减去
# mean = np.mean(data, axis=0)  # 对每列（每个像素位置）求均值
# data_centered = data - mean  # 中心化数据：每列减去均值
#
# # 确保 data_centered 的形状是二维的，打印其形状进行调试
# print(f"Shape of data_centered: {data_centered.shape}")
#
# # 5. 计算协方差矩阵
# cov_matrix = np.cov(data_centered.T)  # 使用转置后的数据计算协方差矩阵
# print(f"Shape of covariance matrix: {cov_matrix.shape}")
#
# # 6. 对协方差矩阵进行特征值分解
# eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
#
# # 7. 对特征值进行排序并选择前n个特征值对应的特征向量
# sorted_indices = np.argsort(eigenvalues)[::-1]  # 排序特征值，降序
# sorted_eigenvectors = eigenvectors[:, sorted_indices]  # 选择对应的特征向量
#
# # 8. 选择主成分数目，例如选择前 10 个主成分
# n_components = 40
# selected_eigenvectors = sorted_eigenvectors[:, :n_components]  # 选择前n个特征向量
#
# # 9. 将数据投影到降维空间（即选择的主成分上）
# projected_data = np.dot(data_centered, selected_eigenvectors)  # 计算投影
#
# # 10. 从降维后的数据恢复原始数据
# reconstructed_data = np.dot(projected_data, selected_eigenvectors.T) + mean  # 恢复数据并加上均值
#
# # 11. 将恢复的数据转换为原始图像的形状
# reconstructed_image = reconstructed_data.reshape(height, width).astype(np.uint8)
#
# # 12. 显示原始图像和降维后的图像
# plt.figure(figsize=(10, 5))
#
# # 显示原始图像
# plt.subplot(1, 2, 1)
# plt.imshow(gray_img, cmap='gray')
# plt.title('Original Grayscale Image')
# plt.axis('off')
#
# # 显示降维后的图像
# plt.subplot(1, 2, 2)
# plt.imshow(reconstructed_image, cmap='gray')
# plt.title(f'Reduced Image (n_components={n_components})')
# plt.axis('off')
#
# plt.show()






                                   # HOG
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
#
#
# # 计算梯度幅值和方向
# def compute_gradient(image):
#     grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # 水平梯度
#     grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # 垂直梯度
#     magnitude = cv2.magnitude(grad_x, grad_y)
#     angle = cv2.phase(grad_x, grad_y, angleInDegrees=True)
#     return magnitude, angle
#
#
# # 计算HOG的直方图
# def compute_histogram(magnitude, angle, cell_size=(8, 8), bins=9):
#     height, width = magnitude.shape
#     cell_histograms = []
#
#     for y in range(0, height, cell_size[1]):
#         for x in range(0, width, cell_size[0]):
#             # 确保cell在图像范围内
#             cell_magnitude = magnitude[y:y + cell_size[1], x:x + cell_size[0]]
#             cell_angle = angle[y:y + cell_size[1], x:x + cell_size[0]]
#
#             if cell_angle.shape[0] != cell_size[1] or cell_angle.shape[1] != cell_size[0]:
#                 continue  # 跳过不足的cell
#
#             # 创建一个直方图
#             hist = np.zeros(bins)
#             for i in range(cell_size[1]):
#                 for j in range(cell_size[0]):
#                     angle_value = cell_angle[i, j] % 180
#                     bin_idx = int(angle_value / (180.0 / bins))
#                     hist[bin_idx] += cell_magnitude[i, j]
#
#             cell_histograms.append(hist)
#
#     return np.array(cell_histograms)
#
#
# # 进行block归一化
# def block_normalization(histograms, height, width, block_size=(2, 2)):
#     blocks = []
#     cells_per_row = width // 8
#     cells_per_col = height // 8
#     for y in range(0, cells_per_col - block_size[1] + 1):
#         for x in range(0, cells_per_row - block_size[0] + 1):
#             block = []
#             for by in range(block_size[1]):
#                 for bx in range(block_size[0]):
#                     idx = (y + by) * cells_per_row + (x + bx)
#                     block.append(histograms[idx])
#
#             block = np.array(block)
#             block = block / np.linalg.norm(block)  # L2归一化
#             blocks.append(block)
#
#     return np.array(blocks)
#
#
# # 提取HOG特征
# def extract_hog_features(image, cell_size=(8, 8), block_size=(2, 2), bins=9):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将RGB图像转换为灰度图
#     magnitude, angle = compute_gradient(gray_image)  # 计算梯度幅值和方向
#     height, width = gray_image.shape
#     histograms = compute_histogram(magnitude, angle, cell_size, bins)  # 计算直方图
#     hog_features = block_normalization(histograms, height, width, block_size)  # 归一化处理
#     return hog_features, histograms, magnitude, angle
#
#
# # 加载图像
# image = cv2.imread('D01961BF87B3C67FF7C4D11DF709ACCE.jpg')
#
# # 确保图像尺寸为572x763
# print("图像尺寸：", image.shape)
#
# # 提取HOG特征
# hog_features, histograms, magnitude, angle = extract_hog_features(image)
#
# # 可视化梯度幅值图像
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(magnitude, cmap='gray')
# plt.title("Gradient Magnitude")
# plt.axis('off')
#
# # 可视化梯度方向图像
# plt.subplot(1, 2, 2)
# plt.imshow(angle, cmap='hsv')  # 使用HSV颜色空间来显示角度
# plt.title("Gradient Direction")
# plt.axis('off')
#
# plt.show()
#
# # 可视化HOG直方图（归一化后的）
# first_histogram = histograms[0]
#
# plt.bar(range(len(first_histogram)), first_histogram, width=1, edgecolor='black')
# plt.title("HOG Histogram (Normalized) for First Cell")
# plt.xlabel("Bin index (gradient direction)")
# plt.ylabel("Magnitude sum")
# plt.show()
#
# # 可视化HOG特征图像
# # 将每个cell的梯度方向按HOG特征可视化，通常是将每个cell的方向绘制在图像上
# cell_size = (8, 8)
# hog_image = np.zeros_like(magnitude)
#
# for y in range(0, magnitude.shape[0], cell_size[1]):
#     for x in range(0, magnitude.shape[1], cell_size[0]):
#         # 计算cell的中心位置
#         cell_magnitude = magnitude[y:y + cell_size[1], x:x + cell_size[0]]
#         cell_angle = angle[y:y + cell_size[1], x:x + cell_size[0]]
#
#         # 计算cell的平均方向
#         angle_value = np.mean(cell_angle) % 180
#         magnitude_value = np.mean(cell_magnitude)
#
#         # 绘制方向箭头
#         arrow_length = magnitude_value / 2
#         arrow_x = int(x + cell_size[0] // 2 + arrow_length * np.cos(np.radians(angle_value)))
#         arrow_y = int(y + cell_size[1] // 2 - arrow_length * np.sin(np.radians(angle_value)))
#         cv2.line(hog_image, (x + cell_size[0] // 2, y + cell_size[1] // 2), (arrow_x, arrow_y), 255, 1)
#
# # 显示HOG图像
# plt.imshow(hog_image, cmap='gray')
# plt.title("HOG Feature Visualization")
# plt.axis('off')
# plt.show()
#
# # 输出HOG特征的长度
# print("HOG特征向量的长度:", len(hog_features))




                                                          # HARRIS

#
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def harris_corner_detection(image, block_size=2, ksize=3, k=0.04, threshold=0.01):
#     # 计算图像的梯度
#     Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
#     Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
#
#     # 计算梯度的乘积
#     Ixx = Ix * Ix
#     Ixy = Ix * Iy
#     Iyy = Iy * Iy
#
#     # 计算每个像素点的自相关矩阵
#     height, width = image.shape
#     R = np.zeros((height, width), dtype=np.float64)
#
#     # 进行窗口卷积计算自相关矩阵的求和
#     for y in range(block_size, height - block_size):
#         for x in range(block_size, width - block_size):
#             # 在当前像素周围的block区域内进行求和
#             Sxx = np.sum(Ixx[y - block_size:y + block_size + 1, x - block_size:x + block_size + 1])
#             Sxy = np.sum(Ixy[y - block_size:y + block_size + 1, x - block_size:x + block_size + 1])
#             Syy = np.sum(Iyy[y - block_size:y + block_size + 1, x - block_size:x + block_size + 1])
#
#             # 计算自相关矩阵的行列式
#             det = Sxx * Syy - Sxy * Sxy
#             # 计算自相关矩阵的迹
#             trace = Sxx + Syy
#             # 计算响应值 R (Harris角点响应值)
#             R[y, x] = det - k * (trace ** 2)
#
#     # 设置阈值，标记角点
#     corners = np.zeros_like(R)
#     corners[R > threshold * R.max()] = 255
#
#     return corners, R
#
#
# # 加载灰度图像
# image = cv2.imread('D01961BF87B3C67FF7C4D11DF709ACCE.jpg', cv2.IMREAD_GRAYSCALE)
#
# # 进行 Harris 角点检测
# corners, response = harris_corner_detection(image)
#
# # 可视化原始图像和检测到的角点
# plt.figure(figsize=(10, 10))
#
# # 原图
# plt.subplot(1, 2, 1)
# plt.imshow(image, cmap='gray')
# plt.title('Original Image')
# plt.axis('off')
#
# # 角点检测结果
# plt.subplot(1, 2, 2)
# plt.imshow(image, cmap='gray')
# plt.title('Harris Corner Detection')
# plt.scatter(np.where(corners == 255)[1], np.where(corners == 255)[0], color='red', s=1)
# plt.axis('off')
#
# plt.show()








                                             # hough transform

#
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 加载图像并转换为灰度图
# image = cv2.imread('D01961BF87B3C67FF7C4D11DF709ACCE.jpg')  # 替换为你图像的路径
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # 应用高斯模糊来去噪声（这对于Hough变换非常重要）
# blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
#
# # 使用Canny边缘检测
# edges = cv2.Canny(blurred_image, 50, 150, apertureSize=3)
#
# # 显示边缘检测的结果
# plt.figure(figsize=(6, 6))
# plt.imshow(edges, cmap='gray')
# plt.title('Edge Detection using Canny')
# plt.axis('off')
# plt.show()
#
# # 使用Hough变换来检测图像中的直线
# # 参数解释：
# # 1. edges: 输入图像，通常是边缘检测后的图像
# # 2. 1: ρ的精度，1个像素
# # 3. np.pi / 180: θ的精度，1度
# # 4. threshold: 阈值，表示最小投票数，低于该值的线会被认为是无效的
# lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
#
# # 绘制检测到的直线
# line_image = np.copy(image)
#
# if lines is not None:
#     for line in lines:
#         rho, theta = line[0]
#         # 计算直线的起始和结束点
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         x1 = int(x0 + 1000 * (-b))
#         y1 = int(y0 + 1000 * (a))
#         x2 = int(x0 - 1000 * (-b))
#         y2 = int(y0 - 1000 * (a))
#
#         # 绘制直线
#         cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
#
# # 显示检测到的直线
# plt.figure(figsize=(6, 6))
# plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
# plt.title('Detected Lines using Hough Transform')
# plt.axis('off')
# plt.show()



                                      # Viola Jones人脸检测
import cv2

# 加载 Haar 特征分类器（Viola-Jones）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 读取图像
image = cv2.imread('帅涛2.0.jpg')

# 将图像转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 检测人脸
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

# 在图像中画出检测到的人脸区域
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# 显示结果
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()





