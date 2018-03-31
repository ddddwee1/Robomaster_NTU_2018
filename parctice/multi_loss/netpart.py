import model as M 
import numpy as np 
import tensorflow as tf 	

def build_model(inp_holder):
	mod = M.Model(inp_holder,[None,None,None,3])
	mod.dwconvLayer(7,4,activation=M.PARAM_LRELU)
	mod.convLayer(3,16,stride=2,activation=M.PARAM_LRELU) #480_ 2x2
	# mod.convLayer(5,16,stride=2,activation=M.PARAM_LRELU) # can add layers to change the size of detection grid
	
	c0 =mod.convLayer(5,16,stride=2,activation=M.PARAM_LRELU) #240_ 4x4
	mod.convLayer(3,32,activation=M.PARAM_LRELU)
	feat0 = mod.convLayer(3,64,activation=M.PARAM_LRELU)
	bias0 = mod.convLayer(1,4)
	mod.set_current(feat0)
	conf0 = mod.convLayer(1,1)

	mod.set_current(c0)
	c1 = mod.convLayer(5,32,stride=2,activation=M.PARAM_LRELU) #120_ 8x8
	mod.convLayer(3,32,activation=M.PARAM_LRELU)
	feat1 = mod.convLayer(3,64,activation=M.PARAM_LRELU)
	bias1 = mod.convLayer(1,4)
	mod.set_current(feat1)
	conf1 = mod.convLayer(1,1)

	mod.set_current(c1)
	c2 = mod.convLayer(5,64,stride=2,activation=M.PARAM_LRELU) #60_  16x16
	mod.convLayer(3,64,activation=M.PARAM_LRELU)
	feat2 = mod.convLayer(3,128,activation=M.PARAM_LRELU)
	bias2 = mod.convLayer(1,4)
	mod.set_current(feat2)
	conf2 = mod.convLayer(1,1)

	mod.set_current(c2)
	mod.convLayer(3,128,stride=2,activation=M.PARAM_LRELU) #30_ 32x32
	mod.convLayer(3,128,activation=M.PARAM_LRELU)
	feat3 = mod.convLayer(3,256,activation=M.PARAM_LRELU)
	bias3 = mod.convLayer(1,4)
	mod.set_current(feat3)
	conf3 = mod.convLayer(1,1)

	return bias0,bias1,bias2,bias3,conf0,conf1,conf2,conf3

def loss_function(bias,conf,b_lab,c_lab):
	bias_loss = tf.reduce_mean(tf.sqaure(bias - b_lab))
	conf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=c_lab,logits=conf))
	return bias_loss + conf_loss

def build_loss(inpholder,b_lab,c_lab):
	b0,b1,b2,b3,c0,c1,c2,c3 = build_model(inpholder)
	loss_functions = []
	loss_functions.append(loss_function(b0,c0,b_lab,c_lab))
	loss_functions.append(loss_function(b1,c1,b_lab,c_lab))
	loss_functions.append(loss_function(b2,c2,b_lab,c_lab))
	loss_functions.append(loss_function(b3,c3,b_lab,c_lab))
	return loss_functions

def get_train_steps(losses):
	train_steps = []
	for loss in losses:
		step = tf.train.AdamOptimizer(0.0001).minimize(loss)
		train_steps.append(step)
	return train_steps

inpholder = tf.placeholder(tf.float32,[None,None,None,3])
b_labholder = tf.placeholder(tf.float32,[None,None,None,4])
c_labholder = tf.placeholder(tf.float32,[None,None,None,1])

loss_functions = build_loss(inpholder,b_labholder,c_labholder)
train_steps = get_train_steps(loss_functions)

# 100x100
# i decide to use the index = 1
# _,loss = sees.run([train_steps[index],loss_functions[index]],feed_dict={sdfasdfsadfsadf})
# print(loss)

with tf.Session() as sess:
	M.loadSess('./model/',sess,init=True)
	tf.summary.FileWriter('./log/',sess.graph)