import cv2
import lane_detection_basic_common_func as basic_common_func
import lane_detection_edge_common_func as edge_common_func

image = basic_common_func.get_image('solidWhiteCurve.jpg')
canny_img = edge_common_func.get_canny_image(image)

cv2.imshow('result', canny_img)
cv2.waitKey(0)