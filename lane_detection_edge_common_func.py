import cv2
import numpy as np
import lane_detection_basic_common_func as basic_common_func

# 흑백이미지 변환
def grayscale(img):     
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 가우시안 필터
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# Canny 알고리즘
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def draw_lines(img, lines, color=[0, 0, 255], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def weighted_img(image, initial_image, alpha=1, beta=1., gamma=0.):
    return cv2.addWeighted(initial_image, alpha, image, beta, gamma)

def get_canny_image(image, low_threshold=70, high_threshold=210, kernel_size=3):
    gray_img = grayscale(image)
    blur_img = gaussian_blur(gray_img, kernel_size)
    canny_img = canny(blur_img, low_threshold, high_threshold)
    return canny_img

def get_hough_image(image, rho=1, theta=np.pi/180, threshold=30, min_line_len=10, max_line_gap=20):
    hough_lines = get_hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap)
    hough_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    draw_lines(hough_img, hough_lines)
    return hough_img

def get_hough_lines(image, rho=1, theta=np.pi/180, threshold=30, min_line_len=10, max_line_gap=20):
    canny_img = get_canny_image(image)
    roi_img = basic_common_func.get_roi_image(canny_img)
    lines = cv2.HoughLinesP(roi_img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines