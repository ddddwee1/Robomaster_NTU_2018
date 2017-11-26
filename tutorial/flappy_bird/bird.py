import sys
sys.path.append('./game/')

import tensorflow as tf 
import model as M 
import numpy as np 
import random
from collections import deque
import cv2


import wrapped_flappy_bird as game

GAME = 'bird'


def build_model(inp):
	mod = M.Model(inp,[None,80,80,4])
	mod.convLayer(5,16,stride=2,activation=M.PARAM_RELU)#40
	mod.convLayer(4,32,stride=2,activation=M.PARAM_RELU)#20
	mod.convLayer(3,64,stride=1,activation=M.PARAM_RELU)
	mod.convLayer(3,64,stride=2,activation=M.PARAM_RELU)#10
	mod.convLayer(3,128,stride=2,activation=M.PARAM_RELU)#5
	mod.flatten()
	mod.fcLayer(2)
	return mod.get_current_layer()

def build_graph():
	imageHolder = tf.placeholder(tf.float32,[None,80,80,4])
	actionHolder = tf.placeholder(tf.float32,[None,2])
	scoreHolder = tf.placeholder(tf.float32)

	output = build_model(imageHolder)
	action_out = tf.reduce_sum(output*actionHolder,axis=1)
	action = tf.argmax(output,axis=1)

	train_step = tf.reduce_mean(tf.square(scoreHolder - action_out))

	return imageHolder,actionHolder,scoreHolder,action,action_out,train_step

EPS = 0.9
EXPLORE = 10000
TRAIN = 30000

def training():
	imageHolder,actionHolder,scoreHolder,action,action_out,train_step = build_graph()

	gstate = game.GameState()
	D = deque()

	#get_initial frame
	do_nothing = np.zeros(2)
	do_nothing[0] = 1
	x_0,r_0,terminal = gstate.frame_step(do_nothing)
	x_t = cv2.cvtColor(cv2.resize(x_t,(80,80)),cv2.COLOR_BGR2GREY)
	_,x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
	s_t = np.stack((x_t,x_t,x_t,x_t),axis=2)


	with tf.Session() as sess:
		M.loadSess('./model/',sess,init=True)
		frame_count = 0
		while True:
			action_array = np.zeros(2)
			if random.random()<EPS:
				action[random.randrange(2)] = 1
			else:
				act = sess.run(action,feed_dict={imageHolder:[s_t]})[0]
				action_array[act] = 1

			x_1,r_1,terminal = gstate.frame_step(action_array)
			x_1 = cv2.cvtColor(cv2.resize(x_1,(80,80)),cv2.COLOR_BGR2GREY)
			_,x_1 = cv2.threshold(x_1,1,255,cv2.THRESH_BINARY)
			s_1 = np.append(x_1.reshape([80,80,1]),s_t[:,:,:3],axis=2)

			D.append()