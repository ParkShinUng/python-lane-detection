import cv2
import numpy as np
import lane_detection_edge_common_func as edge_common_func
import lane_detection_basic_common_func as basic_common_func

cap = basic_common_func.get_image('solidWhiteRight.mp4')

while(cap.isOpened()):
    ret, image = cap.read()
    
    if image is None:
        break
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    hough_img = edge_common_func.get_hough_image(image)

    result = edge_common_func.weighted_img(hough_img, image)
    
    cv2.imshow('result', result)

cap.release()
cv2.destroyAllWindows()