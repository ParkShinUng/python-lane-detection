import os
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESOURCE_IMG_DIR_PATH = os.path.join(PROJECT_ROOT, 'resources', 'images')
RESOURCE_VIDEO_DIR_PATH = os.path.join(PROJECT_ROOT, 'resources', 'video')

def get_image(filename: str):
    if filename is None:
        raise ValueError("filename must be provided")
    
    # Video file
    if ".mp4" in filename:
        file_path = os.path.join(RESOURCE_VIDEO_DIR_PATH, filename)
        return cv2.VideoCapture(file_path)
    # Image file
    else:
        file_path = os.path.join(RESOURCE_IMG_DIR_PATH, filename)
        return cv2.imread(file_path)
    
def get_vertices(image):
    height, width = image.shape[:2]
    vertices = np.array([[(50,height),(width/2-45, height/2+60), (width/2+45, height/2+60), (width-50,height)]], dtype=np.int32)
    return vertices

def get_roi_image(img, color3=(255,255,255), color1=255): # ROI 셋팅
    mask = np.zeros_like(img) # mask = img와 같은 크기의 빈 이미지
    
    # Color 이미지(3채널), 흑백 이미지(1채널) 구분
    color = color3 if len(img.shape) > 2 else color1
        
    # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움 
    vertices = get_vertices(img)
    cv2.fillPoly(mask, vertices, color)
    
    # 이미지와 color로 채워진 ROI를 합침
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image

def get_mark_image(origin_img, roi_img, blue_threshold=200, green_threshold=200, red_threshold=200): # 흰색 차선 찾기
    # BGR 제한 값보다 작으면 검은색으로
    thresholds = (origin_img[:,:,0] < blue_threshold) \
                | (origin_img[:,:,1] < green_threshold) \
                | (origin_img[:,:,2] < red_threshold)
    roi_img[thresholds] = [0, 0, 0]
    mark = roi_img
    return mark