import tensorflow as tf 
import numpy as np 
import model as M 

def build_net(inp):
	mod = M.Model(inp,[None,2])
	mod.fcLayer(200,activation=M.PARAM_RELU)
	mod.fcLayer(200,activation=M.PARAM_RELU)
	mod.fcLayer(4)
	return mod.get_current_layer()

def build_graph():
	envholder = tf.placeholder(tf.float32,[None,2])
	actholder = tf.placeholder(tf.float32,[None,6])
	qholder = tf.placeholder(tf.float32,[None,1])

	values = build_net(envholder)
	q_est = tf.reduce_mean(values*actholder,axis=1,keepdims=True)

	loss = tf.reduce_mean(tf.square(q_est - qholder))

	