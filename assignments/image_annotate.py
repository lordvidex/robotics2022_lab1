import cv2
import numpy as np

img = cv2.imread('assets/map.png')
points = []
def mouse_callback(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
while True:
    if cv2.waitKey(1) & 0xFF == 27:
        break
    # draw circles for each points
    for i in range(len(points)):
        cv2.circle(img, (points[i][0], points[i][1]), 3, (0, 255,0), cv2.FILLED)
    # draw lines to join the points
    for i in range(len(points)):
        if i < len(points) - 1:
            cv2.line(img, (points[i][0], points[i][1]), (points[i+1][0], points[i+1][1]), (255, 0, 0), 2)
    cv2.imshow('map', img)
    cv2.setMouseCallback('map', mouse_callback)
    cv2.waitKey(1)
cv2.destroyAllWindows()