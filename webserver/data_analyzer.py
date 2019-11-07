import collections
import numpy as np
import unicodedata
import re
from tqdm import tqdm
import operator
from random import *
import random
import datetime
import os
import math
import shutil

import queue
import time

import logging, logging.config
import config

import scipy.stats as stats   ###
import collections            ###

import logging, logging.config

import server
import openslide
from openslide_python_fix import _load_image_lessthan_2_29, _load_image_morethan_2_29

def ceildiv(a, b):
	return -(-a // b)

class DataAnalyzer():
	def __init__(self):
		return
	
	def init_project(self, panel_width, patch_size):
		logging.debug("init_project")
		self.patch_size = patch_size
		self.areas = collections.OrderedDict()
		self.level = 0
		self.level_count = 0
		self.patches = {}
		self.panel_width = panel_width
		self.image_ratio = 1

		return

	def set_image_folder(self, path):
		self.image_folder = path

	def get_image_folder(self):
		return self.image_folder

	def set_image_name(self, name):
		self.image_name = name

	def set_level(self, lv):
		self.level = lv

	def set_svs_image(self, path):
		try: 
			self.svs_img = openslide.OpenSlide(os.path.join(path))
		except openslide.OpenSlideError:
			logging.debug("Cannot find file '" + path + "'")
			return
		except openslide.OpenSlideUnsupportedFormatError:
			logging.debug("Slide format not supported. ")
			return

		self.level_count = self.svs_img.level_count
		self.init_patches()

		self.image_ratio = self.svs_img.level_dimensions[self.level][0] / self.panel_width

	def get_rgb_image(self):
		# rect_size = 1000 import logging, logging.config

		dim_x = self.svs_img.level_dimensions[self.level][0]
		dim_y = self.svs_img.level_dimensions[self.level][1]
		logging.debug(dim_x)
		logging.debug(dim_y)

		if (dim_x * dim_y) >= 2**29:
			openslide.lowlevel._load_image = _load_image_morethan_2_29
		else:
			openslide.lowlevel._load_image = _load_image_lessthan_2_29

		img = self.svs_img.read_region((0, 0), self.level, (dim_x, dim_y)).convert("RGB")
		# original_img = slide.read_region((0, 0), 0, (dim_x, dim_y)).convert('RGB')
		# logging.debug("original_img")
		# convert_img = original_img.convert("RGB")
		# logging.debug("convert_img")

		aspect_ratio = dim_y / dim_x
		img = img.resize((self.panel_width, math.floor(self.panel_width*aspect_ratio)))

		# self.image_ratio = dim_x / self.panel_width

		return img

	def delete_label_images(self):

		label_folder = self.image_folder + 'labels/'
		label_folder_list = [name for name in os.listdir(label_folder)
			if os.path.isdir(label_folder)]

		for label in label_folder_list:
			label_path = os.path.join(label_folder, label)
			for file in os.listdir(label_path):
				file_path = os.path.join(label_path, file)

				try:
					if os.path.isfile(file_path):
						os.unlink(file_path)
				except Exception as e:
					print(e)

	def delete_pre_images(self):

		pre_folder = self.image_folder
		pre_folder = os.path.join(pre_folder, 'pre')
		pre_folder = os.path.join(pre_folder, 'images')
		pre_folder = os.path.join(pre_folder, 'selected')

		for file in os.listdir(pre_folder):
			file_path = os.path.join(pre_folder, file)
			try:
				if os.path.isfile(file_path):
					os.unlink(file_path)
			except Exception as e:
				print(e)

	def save_whole_patches(self):
		
		# 1. create folder
		whole_patches_folder = self.image_folder + 'whole_patches/images/'
		if not os.path.exists(whole_patches_folder):
			os.makedirs(whole_patches_folder)

		# 2. get filename
		filename = os.path.splitext(self.image_name)[0]

		# 3. get image size
		x = 0
		y = 0
		w = self.get_image_size()[0]
		# w = math.floor(self.image_ratio * w)
		h = self.get_image_size()[1]
		# h = math.floor(self.image_ratio * h)
		
		patch_size = self.patch_size
		
		pid_top = y // patch_size
		pid_left = x // patch_size
		pid_bottom = ceildiv( (y+h), patch_size ) - 1
		pid_right = ceildiv( (x+w), patch_size ) - 1

		# 3. set and save patches
		for yi in range(pid_top, pid_bottom+1):
			for xi in range(pid_left, pid_right+1):
				
				# save patch file
				pos_x = xi * patch_size - patch_size
				pos_y = yi * patch_size - patch_size
				patch_file_size = self.patch_size * 3
				patch_img = self.svs_img.read_region((pos_x, pos_y), self.level, (patch_file_size, patch_file_size)).convert("RGB")
				patch_img_path = os.path.join(whole_patches_folder, filename + '_' + str(yi) + '_' + str(xi) + '.jpg')
				patch_img.save(patch_img_path)

	def set_label(self, idx, x, y, h, w, lb):

		

		x = math.floor(self.image_ratio * x)
		y = math.floor(self.image_ratio * y)
		h = math.floor(self.image_ratio * h)
		w = math.floor(self.image_ratio * w)
		
		# if idx not in self.areas:
		# 	self.areas[idx] = collections.OrderedDict()
		# self.areas[idx][x] = x
		# self.areas[idx][y] = y
		# self.areas[idx][h] = h
		# self.areas[idx][w] = w
		# self.areas[idx][lb] = lb

		patch_size = self.patch_size

		pid_top = y // patch_size
		pid_left = x // patch_size
		pid_bottom = ceildiv( (y+h), patch_size ) - 1
		pid_right = ceildiv( (x+w), patch_size ) - 1

		# 1. create folder
		if lb == 'selected':
			label_folder = self.image_folder + 'pre/images/' + lb
		else:
			label_folder = self.image_folder + 'labels/' + lb
		if not os.path.exists(label_folder):
			os.makedirs(label_folder)

		# 2. get filename
		filename = os.path.splitext(self.image_name)[0]
		
		# 3. set and save patches
		for yi in range(pid_top, pid_bottom+1):
			for xi in range(pid_left, pid_right+1):
				# set patch
				if lb != 'selected':
					self.patches[self.level][yi][xi]['label'] = lb
				
				# save patch file
				pos_x = xi * patch_size - patch_size
				pos_y = yi * patch_size - patch_size
				patch_file_size = self.patch_size * 3
				patch_img = self.svs_img.read_region((pos_x, pos_y), self.level, (patch_file_size, patch_file_size)).convert("RGB")
				patch_img_path = os.path.join(label_folder, filename + '_' + str(yi) + '_' + str(xi) + '.jpg')
				patch_img.save(patch_img_path)

	def set_patches(self, patches):

		logging.debug(patches.shape)
		h = patches.shape[0]
		w = patches.shape[1]

		# set patches using two-dimensional loop
		for x in range(0, w):
			for y in range(0, h):

				if patches[y, x] == 0:
					label = 'bg' 
				elif patches[y, x] == 1:
					label = 'hp'
				elif patches[y, x] == 2:
					label = 'normal'
				elif patches[y, x] == 3:
					label = 'ta'

				self.patches[self.level][y][x]['label'] = label


	def get_image_size(self):
		return self.svs_img.level_dimensions[self.level]

	def get_patches(self):
		return self.patches

	def init_patches(self):
		logging.debug('init_patches()')
		patch_size = self.patch_size
		
		for l in range(0, self.level_count):
			if l not in self.patches:
				self.patches[l] = {}

			img_width = self.svs_img.level_dimensions[l][0]
			img_height = self.svs_img.level_dimensions[l][1]

			num_patches_hor = ceildiv(img_width, patch_size)
			num_patches_ver = ceildiv(img_height, patch_size)

			for y in range(0, num_patches_ver):
				if y not in self.patches[l]:
					self.patches[l][y] = {}
				for x in range(0, num_patches_hor):
					if x not in self.patches[l][y]:
						self.patches[l][y][x] = {}

					self.patches[l][y][x]['label'] = ''
					self.patches[l][y][x]['pos_x'] = x * patch_size
					self.patches[l][y][x]['pos_y'] = y * patch_size


					