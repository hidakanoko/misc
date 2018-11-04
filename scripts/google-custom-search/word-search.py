#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

def handle_args():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('file', type=argparse.FileType('r', encoding='UTF-8'), help='search word list file', default='words.txt')
	args = parser.parse_args()
	return args.file

def read_word_file(f):
	for line in f.readlines():
		if len(line.strip()) > 0:
			print(line.strip())

wordFile = handle_args()

read_word_file(wordFile)



# curl "https://www.googleapis.com/customsearch/v1?q=%E6%9C%9D%E9%9D%92%E9%BE%8D&cx=006105230996362570803%3Ahsosogl14qk&searchType=image&key=AIzaSyBTkMC0GYE0mug1PbjApPRr2I65enINeWI"

