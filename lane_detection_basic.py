import cv2
import numpy as np
import lane_detection_basic_common_func

image = lane_detection_basic_common_func.get_image('solidWhiteCurve.jpg')

roi_img = lane_detection_basic_common_func.get_roi_image(image, (0, 0, 255)) # vertices에 정한 점들 기준으로 ROI 이미지 생성
copy_roi_img = np.copy(roi_img) # roi_img 복사

mark_img = lane_detection_basic_common_func.get_mark_image(image, copy_roi_img) # 흰색 차선 찾기

# 흰색 차선 검출한 부분을 원본 image에 overlap 하기
color_thresholds = (mark_img[:,:,0] == 0) & (mark_img[:,:,1] == 0) & (mark_img[:,:,2] > 200)
image[color_thresholds] = [0, 0, 255]

cv2.imshow('roi_white', mark_img) # 흰색 차선 추출 결과 출력
cv2.imshow('result', image) # 이미지 출력
cv2.waitKey(0)