#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess

def handle_args():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('file', type=argparse.FileType('r', encoding='UTF-8'), help='search word list file', default='words.txt')
	args = parser.parse_args()
	return args.file

def do_download(f):
	for line in f.readlines():
		if len(line.strip()) > 0:
			cmd = "/usr/local/bin/googleimagesdownload -k '" + line.strip() + "'"
			print("Launching " + cmd)
			subprocess.run(cmd, shell=True)

wordFile = handle_args()

do_download(wordFile)


