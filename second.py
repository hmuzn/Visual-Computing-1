import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

img = cv.imread("moon.png",cv.IMREAD_GRAYSCALE)

def make_cdf(image = img) :
    hist, bins = np.histogram(image.flatten(), 256,[0,256])
    cdf=hist.cumsum()
    cdf_normalized = cdf / cdf.max()
    cdf_display = cdf_normalized * float(hist.max())
    return cdf, cdf_display

cdf, cdf_display = make_cdf(img)
cdf_m = np.ma.masked_equal(cdf,0)
cdf_new = (cdf_m - cdf_m.min()) / cdf_m.max() - cdf_m.min()* 255
cdf_ui8 = cdf_new.astype(np.uint8)

img_he = cv.equalizeHist(img)
cdf_he, cdf_display_he = make_cdf(img_he)

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize = (8,8))
img_ahe = clahe.apply(img)
cdf_ahe, cdf_display_ahe = make_cdf(img_ahe)

plt.subplot(2,3,1)
plt.imshow(img, cmap='gray')
plt.subplot(2,3,4)
plt.plot(cdf_display, color='b')
plt.hist(img.flatten(),256,[0,256],color='r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'),loc='upper left')
plt.title("GrayScale Image")

plt.subplot(2,3,2)
plt.imshow(img_he, cmap='gray')
plt.subplot(2,3,5)
plt.plot(cdf_display_he, color='b')
plt.hist(img_he.flatten(),256,[0,256],color='r')
plt.xlim([0,256])
plt.legend(("cdf","histogram"),loc="upper left")
plt.title("Global Histogram Equalization")

plt.subplot(2,3,3)
plt.imshow(img_ahe, cmap='gray')
plt.subplot(2,3,6)
plt.plot(cdf_display_ahe, color='b')
plt.hist(img_ahe.flatten(),256,[0,256],color='r')
plt.xlim([0,256])
plt.legend(("cdf","histogram"),loc="upper left")
plt.title("Adaptive Histogram Equalization")
plt.show()


    
