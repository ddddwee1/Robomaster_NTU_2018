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
	mod.convLayer(8,32,stride=4,activation=M.PARAM_RELU)#20
	mod.maxpoolLayer(2)
	mod.convLayer(5,64,stride=2,activation=M.PARAM_RELU)#5
	mod.maxpoolLayer(2)
	mod.convLayer(3,64,stride=1,activation=M.PARAM_RELU)#3
	mod.maxpoolLayer(2)
	mod.flatten()
	mod.fcLayer(512,activation=M.PARAM_RELU)
	mod.fcLayer(2)
	return mod.get_current_layer()

def build_graph():
	imageHolder = tf.placeholder(tf.float32,[None,80,80,4])
	actionHolder = tf.placeholder(tf.float32,[None,2])
	scoreHolder = tf.placeholder(tf.float32,[None])

	output = build_model(imageHolder)
	action_out = tf.reduce_sum(output*actionHolder,axis=1)
	action = tf.argmax(output,axis=1)
	next_score = tf.reduce_max(output,axis=1)

	loss = tf.reduce_mean(tf.square(scoreHolder - action_out))

	train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

	return imageHolder,actionHolder,scoreHolder,action,action_out,next_score,train_step,loss

EPS = 0.7
EXPLORE = 10000
TRAIN = 80000
BSIZE = 32
GAMMA = 0.99

def training():
	imageHolder,actionHolder,scoreHolder,action,action_out,next_score,train_step,loss = build_graph()

	gstate = game.GameState()
	D = deque()

	#get_initial frame
	do_nothing = np.zeros(2)
	do_nothing[0] = 1
	x_0,r_0,terminal = gstate.frame_step(do_nothing)
	x_0 = cv2.cvtColor(cv2.resize(x_0,(80,80)),cv2.COLOR_BGR2GRAY)
	_,x_0 = cv2.threshold(x_0,1,255,cv2.THRESH_BINARY)
	s_0 = np.stack((x_0,x_0,x_0,x_0),axis=2)

	epsilon = EPS

	with tf.Session() as sess:
		M.loadSess('./model/',sess,init=True)
		saver = tf.train.Saver()
		frame_count = 0
		while True:
			action_array = np.zeros(2)
			if random.random()<epsilon:
				action_array[0] = 1
			else:
				act,n_score= sess.run([action,next_score],feed_dict={imageHolder:[s_0]})
				act = act[0]
				n_score = n_score[0]
				print('Frame:%d\tEpsilon:%.4f\tQ:%.4f\tAction:%d'%(frame_count,epsilon,n_score,act))
				action_array[act] = 1

			x_1,r_1,terminal = gstate.frame_step(action_array)
			x_1 = cv2.cvtColor(cv2.resize(x_1,(80,80)),cv2.COLOR_BGR2GRAY)
			_,x_1 = cv2.threshold(x_1,1,255,cv2.THRESH_BINARY)
			s_1 = np.append(x_1.reshape([80,80,1]),s_0[:,:,:3],axis=2)

			if len(D)>10000:
				D.popleft()

			# next_r = sess.run(next_score,feed_dict={imageHolder: s_1})

			D.append([s_0,s_1,action_array,r_1,terminal])
			s_0 = s_1

			if frame_count>EXPLORE:
				train_batch = random.sample(D,BSIZE)
				x0batch = [i[0] for i in train_batch]
				a_batch = [i[2] for i in train_batch]
				r_batch = []
				for i in train_batch:
					if i[4]:
						r_batch.append(i[3])
					else:
						scr_next = sess.run(next_score,feed_dict={imageHolder:[i[1]]})[0]
						r_batch.append(i[3]+GAMMA*scr_next)
						# print(r_batch[0])
				_,ls = sess.run([train_step,loss],feed_dict={imageHolder:x0batch, actionHolder:a_batch, scoreHolder:r_batch})
				# print('Loss:',ls)

			if frame_count>EXPLORE and frame_count<TRAIN:
				epsilon -= EPS/(TRAIN-EXPLORE)
				epsilon = max(0.,epsilon)

			frame_count += 1
			if frame_count%10000==0:
				saver.save(sess,'./model/'+str(frame_count)+'.ckpt')

training()