#line_detection.py
import cv2 as cv#
import numpy as np#
import lanes_01 as lanes
from calibrate_01 import *

# To change the readable file, change the path to the files on line 23 and line 7 in calibrate.py 

#33======================== object detection
Conf_threshold = 0.4
NMS_threshold = 0.4
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]
class_name = []
with open('classes.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]
print(class_name)
net = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
#33======================== 
video = cv.VideoCapture('test1.mp4')#
video.set(3, 1920)#1920
video.set(4, 1080)#1080
if not video.isOpened():#
    print('error while opening the video')#
cv.waitKey(1)#
while video.isOpened():#
    _, frame = video.read()#
    cv.namedWindow("Video", cv.WINDOW_NORMAL)
    cv.resizeWindow("Video", 1280, 720)#1280, 720
    video_copy = np.copy(frame)
    #33========================
    classes, scores, boxes = model.detect(video_copy, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_name[classid[0]], score)
        cv.rectangle(video_copy, box, color, 1)
        cv.putText(video_copy, label, (box[0], box[1]-10),
                   cv.FONT_HERSHEY_COMPLEX, 0.5, color, 1) #0.3
#33========================
    try:
        # canny = lanes.canny(frame)
        #Bird = lanes.BEV(canny)
        frame = lanes.canny(frame)
        frame = lanes.mask(frame)
        lines = cv.HoughLinesP(frame, 2, np.pi/180, 100, np.array([()]), minLineLength=20, maxLineGap=5)
        averaged_lines = lanes.average_slope_intercept(frame, lines)
        line_image = lanes.display_lines(video_copy, averaged_lines)
        combo = cv.addWeighted(video_copy, 0.8, line_image, 0.5, 1)
        cv.imshow("Combo", combo)#combo
        # # cv.imshow("Canny", canny)
        # cv.imshow("transformed_frame", Bird)
    except:
        pass
    if cv.waitKey(12) & 0xFF == ord('q'):#
        video.release()#
        cv.destroyAllWindows()#
video.release()#
cv.destroyAllWindows()#