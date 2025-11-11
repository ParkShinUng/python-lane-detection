import cv2
import numpy as np
import lane_detection_basic_common_func as basic_common_func
import lane_detection_edge_common_func as edge_common_func

image = basic_common_func.get_image('solidWhiteCurve.jpg')

canny_img = edge_common_func.get_canny_image(image)
roi_img = basic_common_func.get_roi_image(canny_img)
hough_img = edge_common_func.get_hough_image(roi_img)

result = edge_common_func.weighted_img(hough_img, image) # 원본 이미지에 검출된 선 overlap

cv2.imshow('result', result) # 결과 이미지 출력
cv2.waitKey(0)
