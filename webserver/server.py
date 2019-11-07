# flask & network
from flask import Flask, render_template, request
from flask_api import status
from flask_socketio import SocketIO, send, emit, join_room, leave_room
import requests

# utility
import numpy as np
import json
from datetime import datetime
import logging, logging.config
import netifaces
import time
import signal
import sys
import os

sys.path.insert(0, './model')
from run import Trainer
from model import DenseNet, SimDenseNet2
from utils import cal_l2, ContrastiveLoss, sigmoid
from dataloader import TestImageFolder
# from model import Trainer
# from model import ContrastiveLoss
# from model import DenseNet
# data
# import pickle
# import csv

# user generated
import data_analyzer
import config

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import Subset

import numpy as np
from itertools import combinations

app = Flask(__name__)
app.config['SECRET_KEY'] = 'medical_image_labeling'
app.config['TEMPLATES_AUTO_RELOAD'] = True

socketio = SocketIO(app)
logging.config.fileConfig('logging.conf')

basedir = os.path.abspath(os.path.dirname(__file__))

def signal_handler(sig, frame):
	# da.set_exitFlag(1);
	logging.info('Ctrl+C pressed. Good bye...!!!!');
	sys.exit(0);

signal.signal(signal.SIGINT, signal_handler)

def send_msg(msg_str, msg_param=None):
	emit('server_response', {
		'id': 'message',
		'data': str(msg_str), 
		'param': str(msg_param)
	}, namespace='/title_temp')

def init_project(panel_width, patch_size):
	logging.info("{}> Init project...".format('-'*20))
	time_s = time.time()
	da.init_project(panel_width=panel_width, patch_size=patch_size)
		
	elapsed = time.time() - time_s
	logging.info("{}> Init project complete! Elapsed time: {:3.3f}s".format('-'*20, elapsed))
	return

@app.route('/')
def index():
	logging.info('start')
	return render_template('index.html')

@socketio.on('connect', namespace='/title_temp')
def on_connect():
	logging.info('on_connect()')
	send_msg('Connected')

	# init_project();

@socketio.on('disconnect', namespace='/title_temp')
def on_disconnect():
	logging.info('on_disconnect()')

@socketio.on('request_init_project', namespace='/title_temp')
def on_recv_req_init_project(message):
	logging.debug('on_recv_req_init_project({})'.format(message))
	# socketio.emit('btn_neg_response', {'id': 'btn_neg_res'})   

	panel_width = message['panel_width']
	patch_size = message['patch_size']
	
	init_project(panel_width=panel_width, patch_size=patch_size);

	if da == None:
		logging.warn('da==none')

	# project_info = da.get_project_info()
	# project_info_json = json.dumps(project_info)
	# send_msg('init_project_complete', project_info_json)

@socketio.on('svs_file', namespace='/title_temp')
def on_recv_svs_file(message):
	logging.debug('on_recv_svs_file')

@socketio.on('pass_label_info', namespace='/title_temp')
def on_recv_pass_label_info(message):
	logging.debug('pass_label_info({})'.format(message))

	# da.save_whole_patches()
	
	data = message['data']
	label = message['label']
	da.set_label(data['id'], data['x'], data['y'], data['height'], data['width'], label)
	
	# send response. this is an example
	# send_msg('pass_label_info_received', message)

@socketio.on('delete_label_folder', namespace='/title_temp')
def on_recv_delete_label_folder(message):
	logging.debug('on_recv_delete_label_folder({})'.format(message))

	da.delete_label_images()

@socketio.on('do_train', namespace='/title_temp')
def on_recv_do_train(message):
	logging.debug('on_recv_do_train')

	alpha = float(message['alpha'])

	# get patch image path
	image_dir = da.get_image_folder()

	os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	criterion = nn.CrossEntropyLoss()
	contrasive = ContrastiveLoss(margin=2.0)# .to(device)

	crop_size = 3*256
	transform = transforms.Compose([
						transforms.CenterCrop((crop_size, crop_size)),
						transforms.Resize((256, 256)),
						transforms.ToTensor(),
						transforms.Normalize(mean=(0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])

	model = SimDenseNet2(growthRate=32, depth=24, reduction=0.5, bottleneck=True, nClasses=3).to(device)
	model = nn.DataParallel(model)
	model_dir = os.path.join('', 'static/model/')
	model_path = model_dir + '21_model22.pth'
	model.load_state_dict(torch.load(model_path))

	# 
	model = model.module


	test_dir = os.path.join('', 'static/images/pre/images/')
	trainer = Trainer(model=model, minmax_epochs=(10, 30), alpha=alpha, batch_size=16, test_dir=test_dir)

	# Train with hp1 200 patches
	# trainer.data_dir = image_dir
	# trainer.num_samples = 5
	# trainer.train()

	whole_width = da.get_image_size()[0]
	whole_height = da.get_image_size()[1]
	dir_full_patches = da.get_image_folder() + 'whole_patches/' # drop out /images.
	cmap = trainer.viz_WSI_ft(whole_path=dir_full_patches, whole_wh=(whole_width, whole_height), alpha=alpha, dis_th=0.7) # Use only softmax
	logging.debug(cmap)
	logging.debug('train() complete')

	# set patch information
	da.set_patches(cmap)

	image_size = da.get_image_size()
	patches_info = da.get_patches()
	info = {
		"image_size": image_size,
		"patches_info": patches_info
	}
	info_json = json.dumps(info)
	send_msg('patches_info', info_json)

@socketio.on('request_patch_grid', namespace='/title_temp')
def on_recv_request_patch_grid():
	logging.debug('on_recv_request_patch_grid()')

	image_size = da.get_image_size()
	patches_info = da.get_patches()
	info = {
		"image_size": image_size,
		"patches_info": patches_info
	}
	info_json = json.dumps(info)
	send_msg('patches_info', info_json)


@app.route('/upload_whole_image', methods=['GET','POST'])
def upload_whole_image():
	logging.debug('upload_whole_image')

	whole_image = request.files.get('whole_image')
	logging.debug(whole_image)
	logging.debug('{}'.format(basedir))

	# upload_folder = os.path.join(basedir, '../../miccai_data/')
	upload_folder = os.path.join('', 'static/images/')
	da.set_image_folder(upload_folder)
	da.set_image_name(whole_image.filename)

	# da.delete_label_images()
	da.delete_pre_images()

	upload_path = os.path.join(upload_folder, whole_image.filename)
	whole_image.save(upload_path)
	da.set_svs_image(upload_path)
	da.set_level(0)

	_image_name = whole_image.filename.split('.')[0]

	rgb_path = os.path.join(upload_folder, _image_name+'.jpg')
	logging.debug(rgb_path)

	if os.path.isfile(rgb_path) == False:
		sample_image = da.get_rgb_image()
		sample_image.save(rgb_path)
		# sample_image.save(tiff_path, compression='tiff_lzw')

	return render_template("index.html", sample_image=rgb_path)

# @app.route('/pass_label_info', methods=['POST'])
# def pass_label_info():
#     print('## Receive information')
#     label = request.form['label']
#     return label

@socketio.on_error_default  # handles all namespaces without an explicit error handler
def default_error_handler(e):
	logging.error('Error: ', e)
	send_msg('Error: ' + str(e))

def get_local_ip_address():
	res = 'Cannot find ip address'
	for iface in netifaces.interfaces():
		addresses = netifaces.ifaddresses(iface)
		for address in addresses.keys():
			# print(iface)
			
			if 'addr' not in addresses[address][0].keys():
				continue
			if 'netmask' not in addresses[address][0].keys():
				continue
			if 'broadcast' not in addresses[address][0].keys():
				continue

			if addresses[address][0]['addr'].startswith('127.0.0.1'):
				continue

			if addresses[address][0]['addr'].startswith('172.'):
				continue            

			res = addresses[address][0]['addr']

	return res

def print_welcome_message():
	print('#'*100)
	print(' ')
	print('# Webserver for medical image labeling')
	print('# Server IP/Port: {}:{}'.format(get_local_ip_address(), config.SERVER_PORT))
	print('# The server started at ' + '{:%b %d, %Y}'.format(datetime.now()))
	print('# The server root: ' + app.root_path)
	print(' ')
	print('#'*100)

if __name__ == '__main__':
	print_welcome_message()

	app.use_reloader= True
	SERVER_ROOT = app.root_path
	da = data_analyzer.DataAnalyzer()
	socketio.run(app, host="0.0.0.0", port=config.SERVER_PORT)


@app.route('/to_client/', methods=['POST'])
def to_client():
	print('to_client received')

	content = request.get_json()
	socketio.emit('server_response', {'id': 'to_client', 'data': content})
	return "OK"

@socketio.on('client_event', namespace='/title_temp')
def test_message(message):
	contents = message['data']
	logging.info(contents)
	send_msg(message['data'])

