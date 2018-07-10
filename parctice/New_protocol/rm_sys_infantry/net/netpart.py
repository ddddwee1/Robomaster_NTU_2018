import model as M 
import numpy as np 
import tensorflow as tf 	
import time 

def build_model(inp_holder):
	with tf.variable_scope('MSRPN_v3'):
		mod = M.Model(inp_holder)
		mod.dwconvLayer(7,4,stride=2,activation=M.PARAM_LRELU)
		mod.convLayer(5,16,stride=2,activation=M.PARAM_LRELU) #480_ 2x2	
		# mod.convLayer(5,32,stride=2,activation=M.PARAM_LRELU) #240_ 4x4
		c1 = mod.convLayer(3,32,stride=2,activation=M.PARAM_LRELU) #120_ 8x8
		mod.convLayer(3,32,activation=M.PARAM_LRELU)
		feat1 = mod.convLayer(3,64,activation=M.PARAM_LRELU)
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


inpholder = tf.placeholder(tf.float32,[None,None,None,3])

model_out = build_model(inpholder)
