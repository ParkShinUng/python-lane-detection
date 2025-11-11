import cv2
import numpy as np

# 흑백이미지 변환
def grayscale(img):     
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Canny 알고리즘
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

# 가우시안 필터
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices, color3=(255,255,255), color1=255):
    mask = np.zeros_like(img)
    
    if len(img.shape) > 2:
        color = color3
    else:
        color = color1
        
    cv2.fillPoly(mask, vertices, color)
    
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image

def draw_lines(img, lines, color=[0, 0, 255], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    
    return line_img

def weighted_img(img, initial_img, α=1, β=1., λ=0.): # 두 이미지 operlap 하기
    return cv2.addWeighted(initial_img, α, img, β, λ)

image = cv2.imread('./resources/img/solidWhiteCurve.jpg') # 이미지 읽기
height, width = image.shape[:2] # 이미지 높이, 너비

gray_img = grayscale(image) # 흑백이미지로 변환
    
blur_img = gaussian_blur(gray_img, 3) # Blur 효과
        
canny_img = canny(blur_img, 70, 210) # Canny edge 알고리즘

vertices = np.array([[(50,height),(width/2-45, height/2+60), (width/2+45, height/2+60), (width-50,height)]], dtype=np.int32)
ROI_img = region_of_interest(canny_img, vertices) # ROI 설정

hough_img = hough_lines(ROI_img, 1, 1 * np.pi/180, 30, 10, 20) # 허프 변환

result = weighted_img(hough_img, image) # 원본 이미지에 검출된 선 overlap
cv2.imshow('result',result) # 결과 이미지 출력
cv2.waitKey(0) 
