import requests
import os
import numpy as np
import cv2
import face_recognition
face_detector1=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#Reading the input image now. 
cam = cv2.VideoCapture(0) #index value of webcam
while cam.isOpened():
    state, frame = cam.read() # state tells about image from webcam
    if not state:
        print("no frame")
        break
    blue = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    # img =cv2.imread("vanshaj.jpg")
    faces= face_detector1.detectMultiScale(blue, 1.3, 5)
    for (x,y, w, h) in faces:
        cv2.rectangle(frame, pt1 = (x,y),pt2 = (x+w, y+h), color = (0,255,0),thickness =  3)
        roi_blue = blue[y:y+h,x:x+w]
        roi_color = frame[y:y+h, x:x+w]
    cv2.imshow("cam1", frame)
    if cv2.waitKey(1) == 27: #27 is for exit button. or == ord("q")
        break
cam.release()
cv2.destroyAllWindows()