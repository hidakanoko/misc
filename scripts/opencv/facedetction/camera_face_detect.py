from picamera import PiCamera
from time import sleep
import cv2
import os
import sys

pic_file = './picamera_work.jpg'

if os.path.exists(pic_file):
	os.remove(pic_file)

camera = PiCamera()
for i in range(3, 0, -1):
	print(i)
	sleep(1)
camera.capture(pic_file)

face_classifier = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_classifier)
eye_classifier = "/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml"
eye_cascade = cv2.CascadeClassifier(eye_classifier)

pic = cv2.imread(pic_file)
pic_gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(
	pic_gray,
	scaleFactor=1.1,
	minNeighbors=5,
	minSize=(30, 30))

count = 0;
for (x, y, w, h) in faces:
	count += 1
	cv2.rectangle(pic, (x, y), (x + w, y + h), (255, 0, 0), 2)
	roi_gray = pic_gray[y:y+h, x:x+w]
	roi_color = pic[y:y+h, x:x+w]
	cv2.imwrite("face_{0}.jpg".format(count), roi_color)
	eyes = eye_cascade.detectMultiScale(roi_gray)
	for (ex, ey, ew, eh) in eyes:
		cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)

cv2.imshow("Raspberry pi camera detects {0} faces".format(len(faces)), pic)
cv2.waitKey(0)
cv2.destroyAllWindows()
