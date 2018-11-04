# -*- coding: utf-8 -*-

import cv2
import os

class OpenCVImageDetector:

	def __init__(self):
		self.setCascade('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
		self.__classifier = None

	def setCascade(self, cascade):
		self.__cascade = cascade

	def getClassifier(self):
		if self.__classifier is None:
			self.__classifier = cv2.CascadeClassifier(self.__cascade)
		return self.__classifier

	def _loadImage(self, imgPath):
		return cv2.imread(imgPath)

	def getObjectSavePath(self, imageName, dest, fname_num):
		dotPosition = os.path.basename(imageName).rfind('.')
		fname = os.path.basename(imageName)[0:dotPosition]
		suffix = os.path.basename(imageName)[dotPosition+1:]
		savePath = os.path.join(dest, "{0}_{1}.{2}".format(fname, fname_num, suffix))
		return savePath

	def detectAndSaveObject(self, imgPath, dest):
		print("INFO: Detcting object in " + imgPath)
		img = self._loadImage(imgPath)
		objects = self.detectObjects(img)
		count = 0
		for (x, y, w, h) in objects:
			count += 1
			roi_color = img[y:y+h, x:x+w]
			saveImagePath = self.getObjectSavePath(imgPath, dest, str(count))
			print("INFO: Saving object as Image file in " + saveImagePath)
			cv2.imwrite(saveImagePath, roi_color)

	def detectObjects(self, img):
		ret = []
		cf = self.getClassifier()
		imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		objects = cf.detectMultiScale(
			imgGray,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=(30, 30))
		print("INFO Object(s) found : " + str(len(objects)))
		return objects




