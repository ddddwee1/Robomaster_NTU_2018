import tensorflow as tf 
import model as M 
import numpy as np 
import random

def turret():
	inpholder = tf.placeholder(tf.float32,[None,2])
	mod = M.Model(inpholder,[None,2])
	mod.fcLayer(10,activation=M.PARAM_RELU)
	mod.fcLayer(4)
	# w_hid = M.get_trainable_vars()[0]
	# b_hid = M.get_trainable_vars()[1]
	# w_out = M.get_trainable_vars()[2]
	# b_out = M.get_trainable_vars()[3]
	output = mod.get_current_layer()
	var = M.get_trainable_vars()
	return inpholder,var,output


def mutate(a,rate):
	r,c = a.shape
	b = a.copy()
	for i in r:
		for j in c:
			if random.random()<rate:
				b[i,j] = a[i,j] * (np.power(random.random()-0.5,random.random()*3-1.5))
	return b

def mutate_all(a,rate):
	res = []
	for i in a:
		res.append(mutate(a,rate))
	return res

def cross(a,b,rate):
	r,c = a.shape
	c = a.copy()
	for i in r:
		for j in c:
			if random.random()<rate:
				c[i,j] = b[i,j]
	return c

def cross_all(a,b,rate):
	res = []
	for i in range(len(a)):
		res.append(cross(a[i],b[i],rate))
	return res