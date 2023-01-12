#lanes.py
import cv2 as cv
import numpy as np
from calibrate_01 import *  

import matplotlib.pyplot as plt
height = image.shape[0]
width = image.shape[1]
def canny(frame):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)
    blur = cv.GaussianBlur(frame, (5,5),0)
    return cv.Canny(blur, 50,150) #115, 230   50 150, 150,300

def mask(frame):
    polygons = np.array([(blo, height), (tlo, height-405), (tro, height-405), (bro, height)]) #make polygon
    mask = np.zeros_like(frame) #using the original frame draws an array with zeros
    cv.fillPoly(mask, np.array([polygons], dtype=np.int64), 1024) #draws a polygon into an array with zeros
    masked_image = cv.bitwise_and(frame, mask) #
    return masked_image

# def BEV(frame):
#     cv.circle(frame, (tlo, height-405), 5, (0,0,255), -1)
#     cv.circle(frame, (tro, height-405), 5, (0,0,255), -1)
#     cv.circle(frame, (blo, height), 5, (0,0,255), -1)
#     cv.circle(frame, (bro, height), 5, (0,0,255), -1)

#     pts1 = np.float32([[tlo, height-405],[tro, height-405],[blo, height],[bro, height]])
#     pts2 = np.float32([[0,0],[640,0],[0,480],[640,480]])

#     matrix = cv.getPerspectiveTransform(pts1, pts2)
#     transformed_frame = cv.warpPerspective(frame, matrix, (640,480))
#     return transformed_frame

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):

    left_fit = []
    right_fit = []

    while lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_average)
        return np.array([left_line, right_line])

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        i = 1
        for x1, y1, x2, y2 in lines:
            if i == 1:
                cv.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 8)
                pl = [x1,y1,x2,y2]
                i+=1
            else:
                cv.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 8)
                pts = np.array([[[pl[0], pl[1]], [pl[2], pl[3]], [x2, y2], [x1,y1]]], dtype=np.int32)
                cv.fillPoly(line_image, pts, (0,204,204), lineType=8, shift=0, offset=None)
    return line_image


