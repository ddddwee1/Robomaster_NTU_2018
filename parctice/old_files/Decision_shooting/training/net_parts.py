import tensorflow as tf 
import model as M 
import numpy as np 

def build_actor(inp,scope):
	with tf.variable_scope(scope):
		mod = M.Model(inp,[None,10])
		mod.fcLayer(30,activation=M.PARAM_LRELU)
		mod.fcLayer(40,activation=M.PARAM_LRELU)
		mod.fcLayer(10)
	return mod.get_current_layer()

def build_crit(inp_act,inp_env,scope):
	with tf.variable_scope(scope):
		mod = M.Model(inp_act,[None,10])
		mod.fcLayer(5)
		mod.concat_to_current([inp_env,[None,10]],axis=1)
		mod.fcLayer(30,activation=M.PARAM_LRELU)
		mod.fcLayer(30,activation=M.PARAM_LRELU)
		mod.fcLayer(1)
	return mod.get_current_layer()

def grouping(inp):
	gp0, gp1, gp2, gp3 = tf.split(inp,[3,3,3,1],1)
	return [gp0, gp1 ,gp2 ,gp3]

def activate_group(inp):
	gp0 = tf.nn.softmax(inp[0])
	gp1 = tf.nn.softmax(inp[1])
	gp2 = tf.nn.softmax(inp[2])
	gp3 = tf.sigmoid(inp[3])
	result = tf.concat([gp0,gp1,gp2,gp3],1)
	return result