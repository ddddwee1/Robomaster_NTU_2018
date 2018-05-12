import tensorflow as tf 
import numpy as np 
import model as M 
import cv2 


def verify_net(inp,test):
	with tf.variable_scope('detectionLayers'):
		if not test:
			inp = tf.scan(lambda _,y:tf.image.random_brightness(y,50),inp,initializer=tf.constant(0.0,shape=[24,24,3]))
			inp = tf.scan(lambda _,y:tf.image.random_contrast(y,0.5,2),inp,initializer=tf.constant(0.0,shape=[24,24,3]))
			inp = tf.scan(lambda _,y:tf.image.random_saturation(y,0.5,2),inp,initializer=tf.constant(0.0,shape=[24,24,3]))
		mod = M.Model(inp,[None,24,24,3])
		mod.convLayer(4,32,stride=1,activation=M.PARAM_LRELU)		#32_8x8
		mod.maxpoolLayer(2)
		mod.convLayer(3,64,stride=1,activation=M.PARAM_LRELU)	#16_16x16
		mod.maxpoolLayer(2)
		mod.convLayer(3,128,stride=1,activation=M.PARAM_LRELU)	#8_16x16	
		mod.maxpoolLayer(2)
		mod.convLayer(3,256,stride=1)
		mod.flatten()
		mod.fcLayer(300,activation=M.PARAM_LRELU)
		mod.fcLayer(1)
		return mod.get_current_layer()

def multiloss_RPN(inp,test):
	with tf.variable_scope('mainModel'):
		if not test:
			inp = tf.scan(lambda _,y:tf.image.random_brightness(y,50),inp)
			inp = tf.scan(lambda _,y:tf.image.random_contrast(y,0.5,2),inp)
			inp = tf.scan(lambda _,y:tf.image.random_saturation(y,0.5,2),inp)

		mod = M.Model(inp,[None,None,None,3])
		mod.dwconvLayer(3,5,stride=2,activation=M.PARAM_LRELU)
		mod.convLayer(1,16,activation=M.PARAM_LRELU) #480_ 2x2
		c0 = mod.convLayer(3,32,stride=2,activation=M.PARAM_LRELU) #240_ 4x4
		mod.convLayer(3,48,stride=2,activation=M.PARAM_LRELU)#120_ 8x8
		mod.convLayer(3,64,activation=M.PARAM_LRELU)
		mod.convLayer(3,128,activation=M.PARAM_LRELU)
		feat0 = mod.convLayer(3,192)
		bias0 = mod.convLayer(1,4)
		mod.set_current(feat0)
		conf0 = mod.convLayer(1,1)
		
		mod.set_current(c0)
		c1 =mod.convLayer(5,64,stride=2,activation=M.PARAM_LRELU) #120_  8x8
		mod.convLayer(3,128,stride=2,activation=M.PARAM_LRELU) #60_ 16x16
		mod.convLayer(3,128,activation=M.PARAM_LRELU)
		feat1 = mod.convLayer(3,256)
		bias1 = mod.convLayer(1,4)
		mod.set_current(feat1)
		conf1 = mod.convLayer(1,1)

		mod.set_current(c1)
		mod.convLayer(4,128,stride=2,activation=M.PARAM_LRELU)#60_ 16x16
		mod.convLayer(3,128,stride=2,activation=M.PARAM_LRELU)#30_ 32x32
		mod.convLayer(3,192,activation=M.PARAM_LRELU)
		feat2 = mod.convLayer(3,256)
		bias2 = mod.convLayer(1,4)
		mod.set_current(feat2)
		conf2 = mod.convLayer(1,1)


	return bias0[0],bias1[0],bias2[0],conf0[0],conf1[0],conf2[0]

#loss_functions = build_loss(inpholder,b_labholder,c_labholder)
#train_steps = get_train_steps(loss_functions)

# 100x100
# i decide to use the index = 1
# _,loss = sees.run([train_steps[index],loss_functions[index]],feed_dict={sdfasdfsadfsadf})
# print(loss)

