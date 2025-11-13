import cv2
import numpy as np
import lane_detection_basic_common_func as basic_common_func
import lane_detection_edge_common_func as edge_common_func

cap = basic_common_func.get_image('slope_test.mp4')

while(cap.isOpened()):
    ret, image = cap.read()
    
    if image is None:
        break
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    hough_lines = edge_common_func.get_hough_lines(image)

    line_arr = np.squeeze(hough_lines)

    # 기울기 산출
    slope_degree = (np.arctan2(line_arr[:,1] - line_arr[:,3], line_arr[:,0] - line_arr[:,2]) * 180) / np.pi

    # 수평 기울기 제한
    holizontal_mask = np.abs(slope_degree) < 160
    line_arr = line_arr[holizontal_mask]
    slope_degree = slope_degree[holizontal_mask]

    # 수직 기울기 제한
    vertical_mask   = np.abs(slope_degree) > 95
    line_arr = line_arr[vertical_mask]
    slope_degree = slope_degree[vertical_mask]

    # 필터링된 직선 제거
    L_lines, R_lines = line_arr[(slope_degree > 0), :], line_arr[(slope_degree < 0), :]
    L_lines, R_lines = L_lines[:, None], R_lines[:, None]

    # 대표선 Draw
    left_fit_line = edge_common_func.get_fitline(image, L_lines)
    right_fit_line = edge_common_func.get_fitline(image, R_lines)

    filtered_image_arr = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    edge_common_func.draw_fit_line(filtered_image_arr, left_fit_line)
    edge_common_func.draw_fit_line(filtered_image_arr, right_fit_line)

    result = edge_common_func.weighted_img(filtered_image_arr, image) # 원본 이미지에 검출된 선 overlap

    cv2.imshow('result', result)

cap.release()
cv2.destroyAllWindows()