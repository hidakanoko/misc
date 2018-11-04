# -*- coding: utf-8 -*-

from enum import Enum
import subprocess

class GidDownloader:
	"""
	Download images with google-images-downloader.
	https://github.com/hardikvasa/google-images-download
	"""

	def __init__(self):
		self.setTarget('./target')
		self.setType(None)
		self.setLimit(0)

	def setTarget(self, target):
		self.__target = target

	def setType(self, type):
		if type is None:
			self.__type = None
		else:
			try:
				self.__type = GidImageType(type)
			except ValueError as err:
				print("ERROR: Invalid value for type: {0}".format(err))
				exit(1)

	def setLimit(self, limit):
		self.__limit = limit

	def doDownload(self, keywords):
		for keyword in keywords:
			cmd = self.__buildCmd(keyword)
			print("INFO: Starting image download \"{0}\"".format(cmd))
			subprocess.run(cmd, shell=True)

	def __buildCmd(self, keyword):
		cmd = "/usr/local/bin/googleimagesdownload -k '" + keyword.strip() + "'"
		if self.__limit > 0:
			cmd += " --limit " + str(self.__limit)
		if self.__type is not None:
			cmd += " --type " + self.__type.value
		if self.__target is not None:
			cmd += " --output_directory " + self.__target
		return cmd

class GidImageType(Enum):
	FACE = 'face'
	PHOTO = 'photo'
	CLIPART = 'clip-art'
	LINEDRAWING = 'line-drawing'
	ANIMATED = 'animated'


