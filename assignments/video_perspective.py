import cv2
import numpy as np

# read the video
capture = cv2.VideoCapture('assets/road.mp4')
if not capture.isOpened():
    print("Cannot open video")
    exit()

while True:
    ret, frame = capture.read()
    original_h, original_w = frame.shape[:2]

    # width, height = 250, 350
    # pts1 = np.float32([[5, 202],[original_w, 203],[13, original_h], [original_w, original_h]])
    # pts2 = np.float32([[0, 0], [original_w, 0], [0, original_h], [original_w, original_h]])
    
    # matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # imgOutput = cv2.warpPerspective(frame, matrix, (width, height))
    if ret:
        cv2.imshow('frame', frame)
        key = cv2.waitKey(60)
    else:
        break

capture.release()
cv2.destroyAllWindows()