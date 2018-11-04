#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

from GoogleImagesDownload import GidDownloader
from OpenCV import OpenCVImageDetector

def handle_args():
	parser = argparse.ArgumentParser(
		prog='ImageCollector',
		description='Collect images searched by Google and optionally detect faces.')

	parser.add_argument('-k', '--keywords', type=str, action='append', required=True)
	parser.add_argument('-d', '--dest', type=str, default='/tmp/')
	parser.add_argument('-s', '--skipDownload', action='store_const', const=True, default=False)
	parser.add_argument('-o', '--detectObject', action='store_const', const=True, default=False)
	parser.add_argument('-t', '--searchImageType', type=str, default=None)
	parser.add_argument('-l', '--downloadLimit', type=int, choices=range(1,101), default=100)

	args = parser.parse_args()
	return args

def getDownloader(args):
	downloader = GidDownloader()
	downloader.setTarget(args.dest)
	downloader.setLimit(args.downloadLimit)
	downloader.setType(args.searchImageType)
	return downloader

def getObjectDetector(args):
	return OpenCVImageDetector()

def getImagesInDir(target):
	parent, dirs, files = os.walk(target)
	for image in files:
		print(image)

def isSupportedFormat(path):
	# all supported format: https://docs.opencv.org/3.4.3/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
	return path.endswith('.jpeg') \
		 or path.endswith('jpg') \
		 or path.endswith('png') \
		 or path.endswith('bmp')

args = handle_args()

if not args.skipDownload:
	downloader = getDownloader(args)
	downloader.doDownload(args.keywords)

if args.detectObject:
	detector = getObjectDetector(args)
	for dir in os.listdir(args.dest):
		dirpath = os.path.join(args.dest, dir)
		if os.path.isdir(dirpath):
			for img in os.listdir(dirpath):
				imgpath = os.path.join(dirpath, img)
				objectDest = os.path.join(dirpath, "objects")
				if not os.path.exists(objectDest):
					os.makedirs(objectDest)
				if isSupportedFormat(imgpath):
					detector.detectAndSaveObject(imgpath, objectDest)







