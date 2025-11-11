import cv2
import numpy as np
import lane_detection_basic_common_func as basic_common_func
import lane_detection_edge_common_func as edge_common_func

image = basic_common_func.get_image('slope_test.jpg')
hough_img = edge_common_func.get_hough_image(image)

line_arr = np.squeeze(hough_img)

# 기울기 산출
slope_degree = (np.arctan2(line_arr[:,1] - line_arr[:,3], line_arr[:,0] - line_arr[:,2]) * 180) / np.pi

print(line_arr.shape, slope_degree.shape)
holizontal_mask = (np.abs(slope_degree) < 160)
vertical_mask   = (np.abs(slope_degree) > 95)
    
# 수평 기울기 제한
line_arr = line_arr[holizontal_mask]
slope_degree = slope_degree[holizontal_mask]

# 수직 기울기 제한
line_arr = line_arr[vertical_mask]
slope_degree = slope_degree[vertical_mask]

# 필터링된 직선 제거
L_lines, R_lines = line_arr[(slope_degree > 0), :], line_arr[(slope_degree < 0), :]
L_lines, R_lines = L_lines[:, None], R_lines[:, None]
filtered_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

# 직선 Draw
edge_common_func.draw_lines(filtered_image, L_lines)
edge_common_func.draw_lines(filtered_image, R_lines)

result = edge_common_func.weighted_img(hough_img, image) # 원본 이미지에 검출된 선 overlap

cv2.imshow('result', result) # 결과 이미지 출력
cv2.waitKey(0)
