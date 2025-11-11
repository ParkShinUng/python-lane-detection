import cv2
import numpy as np
import common_func

# 흑백이미지 변환
def grayscale(img):     
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Canny 알고리즘
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

# 가우시안 필터
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

image = common_func.get_image('solidWhiteCurve.jpg')
height, width = image.shape[:2]

gray_img = grayscale(image)

blur_img = gaussian_blur(gray_img, 3)

canny_img = canny(blur_img, 70, 210)

cv2.imshow('result', canny_img)
cv2.waitKey(0)