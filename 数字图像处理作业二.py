import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from PIL import Image
from torch.fft import fft
import skimage
image = cv2.imread('266CABC327FEBF78D8FBC56345EB0AF0.jpg',cv2.IMREAD_GRAYSCALE)
# cv2.imshow('origin',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# plt.hist(image.ravel(), 256)
# plt.title('origin')
# plt.show()
# plt.subplot(222)
# 添加高斯噪声
def gaussian_noise(image, mean, sigma):
    """
    添加高斯噪声
    :param image:原图
    :param mean:均值
    :param sigma:标准差 值越大，噪声越多
    :return:噪声处理后的图片
    """
    image = np.asarray(image / 255, dtype=np.float32)  # 图片灰度标准化
    noise = np.random.normal(mean, sigma, image.shape).astype(dtype=np.float32)  # 产生高斯噪声
    output = image + noise  # 将噪声和图片叠加
    output = np.clip(output, 0, 1)
    output = np.uint8(output * 255)
    return output

# 添加椒盐噪声

def add_salt_and_pepper_noise(image, amount):
    # 复制图像以避免修改原始图像
    noisy_image = np.copy(image)

    # 获取图像的高度和宽度
    height, width = image.shape[:2]
    num_salt = int(amount * height * width * 0.5)  # 一半的噪声为盐（白色）
    num_pepper = int(amount * height * width * 0.5)  # 一半的噪声为椒（黑色）

    # 添加白色噪音点（盐）
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1]] = 255

    # 添加黑色噪音点（椒）
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1]] = 0

    return noisy_image

# 均值噪声
def uniform_noise(image, mean, sigma):

    a = 2 * mean - np.sqrt(12 * sigma)  # a = -14.64
    b = 2 * mean + np.sqrt(12 * sigma)  # b = 54.64
    noiseUniform = np.random.uniform(a, b, image.shape)
    imgUniformNoise = image + noiseUniform
    imgUniformNoise = np.uint8(cv2.normalize(imgUniformNoise, None, 0, 255, cv2.NORM_MINMAX))  # 归一化为 [0,255]
    return imgUniformNoise

test1=gaussian_noise(image,0.1,0.1)
# cv2.imshow('gaosi',test1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# plt.figure()  # 新建一个图像
# plt.subplot(221)
# plt.hist(test1.ravel(), 256)
# plt.title('gaosi')
# plt.show()
# plt.subplot(222)
# hist = cv2.calcHist([test1], [0], None, [256], [0, 256])
# plt.plot(hist)
# plt.show()

test2=add_salt_and_pepper_noise(image,0.01)
# cv2.imshow('salt',test2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# plt.figure()  # 新建一个图像
# plt.subplot(221)
# plt.hist(test2.ravel(), 256)
# plt.title('salt')
# plt.show()

test3=uniform_noise(image,10,100)
# cv2.imshow('uniform',test3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# plt.figure()  # 新建一个图像
# plt.subplot(221)
# plt.hist(test3.ravel(), 256)
# plt.title('uniform')
# plt.show()



# 均值滤波器


# 算术平均滤波器


def arithmetic_mean_filter(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image

kernel_size = 3
result = arithmetic_mean_filter(test2, kernel_size)
cv2.imshow('Original Image', test2)
cv2.imshow('suanshujunzhi Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 几何均值滤波器
def geometric_mean_filter(image, kernel_size):
    height, width = image.shape[:2]
    pad = kernel_size // 2
    new_image = np.zeros((height, width), dtype=np.float32)
    image = image.astype(np.float32)
    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            product = 1
            for m in range(i - pad, i + pad + 1):
                for n in range(j - pad, j + pad + 1):
                    product *= image[m, n]
            new_image[i, j] = product ** (1.0 / (kernel_size * kernel_size))
    new_image = np.clip(new_image, 0, 255).astype(np.uint8)
    return new_image

kernel_size = 3
filtered_image = geometric_mean_filter(test2, kernel_size)
cv2.imshow('Original Image', test2)
cv2.imshow('jihejunzhi Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#                      反谐波平均滤波器
# 9.11: 反谐波平均滤波器 (Inv-harmonic mean filter)
img=add_salt_and_pepper_noise(image,0.01)
img_h = img.shape[0]
img_w = img.shape[1]

m, n = 3, 3
order = m * n
kernalMean = np.ones((m,n), np.float32)  # 生成盒式核

hPad = int((m-1) / 2)
wPad = int((n-1) / 2)
imgPad = np.pad(img.copy(), ((hPad, m-hPad-1), (wPad, n-wPad-1)), mode="edge")

Q = 1.5  # 反谐波平均滤波器 阶数
epsilon = 1e-8
imgHarMean = img.copy()
imgInvHarMean = img.copy()
for i in range(hPad, img_h + hPad):
    for j in range(wPad, img_w + wPad):
            # 谐波平均滤波器 (Harmonic mean filter)
        sumTemp = np.sum(1.0 / (imgPad[i-hPad:i+hPad+1, j-wPad:j+wPad+1] + epsilon))
        imgHarMean[i-hPad][j-wPad] = order / sumTemp
            # 反谐波平均滤波器 (Inv-harmonic mean filter)
        temp = imgPad[i-hPad:i+hPad+1, j-wPad:j+wPad+1] + epsilon
        imgInvHarMean[i-hPad][j-wPad] = np.sum(np.power(temp, (Q+1))) / np.sum(np.power(temp, Q) + epsilon)

plt.figure(figsize=(9, 6))
plt.subplot(131), plt.axis('off'), plt.title("Original")
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.subplot(132), plt.axis('off'), plt.title("Harmonic mean filter")
plt.imshow(imgHarMean, cmap='gray', vmin=0, vmax=255)
plt.subplot(133), plt.axis('off'), plt.title("Invert harmonic mean")
plt.imshow(imgInvHarMean, cmap='gray', vmin=0, vmax=255)
plt.tight_layout()
plt.show()


# 修正阿尔法均值滤波器


def median_filter(image, kernel):
    height, width = image.shape[:2]
    m, n = kernel.shape[:2]

    padding_h = int((m - 1) / 2)
    padding_w = int((n - 1) / 2)

    # 这样的填充方式，可以奇数核或者偶数核都能正确填充
    image_pad = np.pad(image, ((padding_h, m - 1 - padding_h),
                               (padding_w, n - 1 - padding_w)), mode="edge")

    image_result = np.zeros(image.shape)
    for i in range(height):
        for j in range(width):
            temp = image_pad[i:i + m, j:j + n]
            image_result[i, j] = np.median(temp)
    return image_result

#修正的阿尔法均值滤波
def modified_alpha_mean(image, kernel, d=0):

    height, width = image.shape[:2]
    m, n = kernel.shape[:2]

    padding_h = int((m - 1) / 2)
    padding_w = int((n - 1) / 2)

    # 这样的填充方式，可以奇数核或者偶数核都能正确填充
    image_pad = np.pad(image, ((padding_h, m - 1 - padding_h),
                               (padding_w, n - 1 - padding_w)), mode="edge")

    img_result = np.zeros(image.shape)
    for i in range(height):
        for j in range(width):
            temp = np.sum(image_pad[i:i + m, j:j + n] * 1)
            img_result[i, j] = temp / (m * n - d)
    return img_result

kernel = np.ones([5, 5])
img_median = median_filter(test2, kernel=kernel)
img_modified_alpha = modified_alpha_mean(test2, kernel=kernel, d=20)

plt.figure(figsize=(8,5))
plt.imshow(image, cmap='gray'), plt.title('origin')
plt.show()
plt.imshow(test2, cmap='gray'), plt.title('junyun')
plt.show()
plt.imshow(img_median, cmap='gray'), plt.title('median')
plt.show()
plt.imshow(img_modified_alpha, cmap='gray'), plt.title('alphatrimmed')
plt.show()



# 运动模糊




# 9.21: 约束最小二乘方滤波
def getMotionDsf(shape, angle, dist):
    xCenter = (shape[0] - 1) / 2
    yCenter = (shape[1] - 1) / 2
    sinVal = np.sin(angle * np.pi / 180)
    cosVal = np.cos(angle * np.pi / 180)
    PSF = np.zeros(shape)  # 点扩散函数
    for i in range(dist):  # 将对应角度上motion_dis个点置成1
        xOffset = round(sinVal * i)
        yOffset = round(cosVal * i)
        PSF[int(xCenter - xOffset), int(yCenter + yOffset)] = 1
    return PSF / PSF.sum()  # 归一化

def makeBlurred(image, PSF, eps):  # 对图片进行运动模糊
    fftImg = np.fft.fft2(image)  # 进行二维数组的傅里叶变换
    fftPSF = np.fft.fft2(PSF) + eps
    fftBlur = np.fft.ifft2(fftImg * fftPSF)
    fftBlur = np.abs(np.fft.fftshift(fftBlur))
    return fftBlur

def wienerFilter(input, PSF, eps, K=0.01):  # 维纳滤波，K=0.01
    fftImg = np.fft.fft2(input)
    fftPSF = np.fft.fft2(PSF) + eps
    fftWiener = np.conj(fftPSF) / (np.abs(fftPSF)**2 + K)
    imgWienerFilter = np.fft.ifft2(fftImg * fftWiener)
    imgWienerFilter = np.abs(np.fft.fftshift(imgWienerFilter))
    return imgWienerFilter

def getPuv(image):
    h, w = image.shape[:2]
    hPad, wPad = h - 3, w - 3
    pxy = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    pxyPad = np.pad(pxy, ((hPad//2, hPad - hPad//2), (wPad//2, wPad - wPad//2)), mode='constant')
    fftPuv = np.fft.fft2(pxyPad)
    return fftPuv

def leastSquareFilter(image, PSF, eps, gamma=0.01):  # 约束最小二乘方滤波
    fftImg = np.fft.fft2(image)
    fftPSF = np.fft.fft2(PSF)
    conj = fftPSF.conj()
    fftPuv = getPuv(image)
    # absConj = np.abs(fftPSF) ** 2
    Huv = conj / (np.abs(fftPSF)**2 + gamma * (np.abs(fftPuv)**2))
    ifftImg = np.fft.ifft2(fftImg * Huv)
    ifftShift = np.abs(np.fft.fftshift(ifftImg))
    imgLSFilter = np.uint8(cv2.normalize(np.abs(ifftShift), None, 0, 255, cv2.NORM_MINMAX))  # 归一化为 [0,255]
    return imgLSFilter


hImg, wImg = image.shape[:2]

    # 带有噪声的运动模糊
PSF = getMotionDsf((hImg, wImg), 45, 100)  # 运动模糊函数
imgBlurred = np.abs(makeBlurred(image, PSF, 1e-6))  # 生成不含噪声的运动模糊图像
# plt.figure(figsize=(9, 7))
# plt.imshow(imgBlurred,'gray')
# plt.show()

scale = 0.01  # 噪声方差
imgBlurNoisy =gaussian_noise(imgBlurred,0.1,0.04) #gaosimohu
# plt.figure(figsize=(9, 7))
# plt.imshow(imgBlurNoisy,'gray')
# plt.show()

imgWienerFilter = wienerFilter(imgBlurNoisy, PSF, scale, K=0.01)  # 对含有噪声的模糊图像进行维纳滤波
imgLSFilter = leastSquareFilter(imgBlurNoisy, PSF, scale, gamma=0.01)  # 约束最小二乘方滤波


plt.figure(figsize=(9, 7))
plt.subplot(231), plt.title("blurred image (dev=0.01)"), plt.axis('off'), plt.imshow(imgBlurNoisy, 'gray')
plt.subplot(232), plt.title("Wiener filter"), plt.axis('off'), plt.imshow(imgWienerFilter, 'gray')
plt.subplot(233), plt.title("least square filter"), plt.axis('off'), plt.imshow(imgLSFilter, 'gray')

# scale = 0.1  # 噪声方差
imgBlurNoisy = gaussian_noise(imgBlurred,0.1,0.04) # 带有噪声的运动模糊
# plt.figure(figsize=(9, 7))
# plt.imshow(imgBlurNoisy,'gray')
# plt.show()
imgWienerFilter = wienerFilter(imgBlurNoisy, PSF, scale, K=0.02)  # 维纳滤波
imgLSFilter = leastSquareFilter(imgBlurNoisy, PSF, scale, gamma=0.1)  # 约束最小二乘方滤波

plt.subplot(234), plt.title("blurred image (dev=0.01)"), plt.axis('off'), plt.imshow(imgBlurNoisy, 'gray')
plt.subplot(235), plt.title("Wiener filter"), plt.axis('off'), plt.imshow(imgWienerFilter, 'gray')
plt.subplot(236), plt.title("least square filter"), plt.axis('off'), plt.imshow(imgLSFilter, 'gray')
plt.tight_layout()
plt.show()