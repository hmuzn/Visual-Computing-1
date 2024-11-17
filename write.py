import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

# 이미지 불러오기
img_gray = cv.imread("write.png", cv.IMREAD_GRAYSCALE)

# 이미지에 로컬 오츠 임계값 적용
def local_otsu(image, block_size):
    height, width = image.shape
    result_image = np.zeros((height, width), dtype=np.uint8)
    half_block = block_size // 2
    for i in range(half_block, height - half_block):
        for j in range(half_block, width - half_block):
            block = image[i - half_block:i + half_block + 1, j - half_block:j + half_block + 1]
            # 로컬 오츠 임계값 계산
            _, thresholded_block = cv.threshold(block, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            
            # 결과 이미지에 임계값 적용
            result_image[i, j] = thresholded_block[half_block, half_block]
    return result_image

block_size = 21
img_th1 = local_otsu(img_gray, block_size)

# global, adaptive
th2v, img_th2 = cv.threshold(img_gray, 70, 225, cv.THRESH_BINARY + cv.THRESH_OTSU)
img_th3 = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)

print(img_gray.shape)

plt.subplot(2, 2, 1)
plt.imshow(img_gray, cmap='gray')
plt.title(f"RGB Image")
plt.subplot(2, 2, 2)
plt.imshow(img_th1, cmap="gray")
plt.title(f"Local Otsu Thresholding")
plt.subplot(2, 2, 3)
plt.imshow(img_th2, cmap='gray')
plt.title(f"Global Otsu Threshodling(v={th2v})")
plt.subplot(2, 2, 4)
plt.imshow(img_th3, cmap='gray')
plt.title("Adaptive Mean Thresholding")

plt.show()


