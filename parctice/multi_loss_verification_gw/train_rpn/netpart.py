import model as M 
import numpy as np 
import tensorflow as tf 	
import time 

def build_model(inp_holder):
	with tf.variable_scope('MSRPN_v2'):
		inp_holder = tf.image.random_saturation(inp_holder,lower=0.5,upper=1.5)
		inp_holder = tf.image.random_contrast(inp_holder,lower=0.5,upper=2.)
		# inp_holder = tf.image.random_saturation(inp_holder,lower=0.5,upper=1.5)
		inp_holder = tf.image.random_brightness(inp_holder,50)
		mod = M.Model(inp_holder)
		mod.dwconvLayer(7,4,stride=2,activation=M.PARAM_LRELU)
		mod.convLayer(3,16,activation=M.PARAM_LRELU) #480_ 2x2	
		mod.convLayer(5,32,stride=2,activation=M.PARAM_LRELU) #240_ 4x4
		c1 = mod.convLayer(5,32,stride=2,activation=M.PARAM_LRELU) #120_ 8x8
		mod.convLayer(3,64,activation=M.PARAM_LRELU)
		feat1 = mod.convLayer(3,128,activation=M.PARAM_LRELU)
		bias1 = mod.convLayer(1,4)
		mod.set_current(feat1)
		conf1 = mod.convLayer(1,1)
		
		mod.set_current(c1)
		mod.convLayer(3,64,stride=2,activation=M.PARAM_LRELU) #60_  16x16
		c3 = mod.convLayer(3,128,stride=2,activation=M.PARAM_LRELU) #30_ 32x32
		feat3 = mod.convLayer(3,128,activation=M.PARAM_LRELU)
		bias3 = mod.convLayer(1,4)
		mod.set_current(feat3)
		conf3 = mod.convLayer(1,1)

		mod.set_current(c3)
		mod.convLayer(3,128,stride=2,activation=M.PARAM_LRELU)# 64x64
		feat4 = mod.convLayer(3,128,stride=2,activation=M.PARAM_LRELU) #128x128
		bias4 = mod.convLayer(1,4)
		mod.set_current(feat4)
		conf4 = mod.convLayer(1,1)
	return bias1,bias3,bias4,conf1,conf3,conf4

def loss_function(bias,conf,b_lab,c_lab,coef=1.):
	bias_loss = tf.reduce_sum(tf.reduce_mean(tf.square(bias - b_lab)*c_lab,axis=0))
	conf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=c_lab,logits=conf))
	return (bias_loss  + conf_loss*10)*coef

def build_loss(model_out,b_lab,c_lab):
	b0,b1,b2,c0,c1,c2 = model_out
	loss_functions = []
	loss_functions.append(loss_function(b0,c0,b_lab,c_lab,11.3))
	loss_functions.append(loss_function(b1,c1,b_lab,c_lab))
	loss_functions.append(loss_function(b2,c2,b_lab,c_lab,1./11.3))
	return loss_functions

def get_train_steps(losses):
	train_steps = []
	for loss in losses:
		step = tf.train.AdamOptimizer(0.00001).minimize(loss)
		train_steps.append(step)
	return train_steps

inpholder = tf.placeholder(tf.float32,[None,None,None,3])
b_labholder = tf.placeholder(tf.float32,[None,None,None,4])
c_labholder = tf.placeholder(tf.float32,[None,None,None,1])

model_out = build_model(inpholder)
loss_functions = build_loss(model_out,b_labholder,c_labholder)
train_steps = get_train_steps(loss_functions)
