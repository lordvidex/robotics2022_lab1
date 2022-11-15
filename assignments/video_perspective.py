import cv2
import numpy as np
points = []
def mouse_drawing(event, x, y, flags, _):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(points)
# read the video
capture = cv2.VideoCapture('assets/road.mp4')
if not capture.isOpened():
    print("Cannot open video")
    exit()

while True:
    ret, frame = capture.read()
    width, height = 250, 350
    pts1 = np.float32([[380, 171],[387, 173], [13, 353], [608, 353]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    # pts1 = np.float32([[5, 202],[original_w, 203],[13, original_h], [original_w, original_h]])
    # pts2 = np.float32([[0, 0], [original_w, 0], [0, original_h], [original_w, original_h]])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(frame, matrix, (width, height))
    if ret:
        # original_h, original_w = frame.shape[:2]
        cv2.imshow('frame', imgOutput)
        cv2.setMouseCallback('frame', mouse_drawing)
        key = cv2.waitKey(10)
    else:
        break

capture.release()
cv2.destroyAllWindows()
print(points)