import tensorflow as tf 
import model as M 
import numpy as np 
import random

class turret():
	def __init__(self):
		self.inpholder = tf.placeholder(tf.float32,[None,3])
		mod = M.Model(self.inpholder,[None,2])
		mod.fcLayer(10)
		mod.fcLayer(2)
		self.w_hid = M.get_trainable_vars()[0]
		self.b_hid = M.get_trainable_vars()[1]
		self.w_out = M.get_trainable_vars()[2]
		self.b_out = M.get_trainable_vars()[3]
		self.output = mod.get_current_layer()
		self.var = M.get_trainable_vars()


def mutate(a,rate):
	r,c = a.shape
	b = a.copy()
	for i in r:
		for j in c:
			if random.random()<rate:
				b[i,j] = a[i,j] * (np.power(random.random()-0.5,random.random()*3-1.5))
	return b

def cross(a,b,rate):
	r,c = a.shape
	c = a.copy()
	for i in r:
		for j in c:
			if random.random()<rate:
				c[i,j] = b[i,j]
	return c

