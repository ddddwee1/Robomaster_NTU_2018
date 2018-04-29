import tensorflow as tf 
import numpy as np 

l_num = 0

###########################################################
#define weight and bias initialization

def weight(shape):
	return tf.get_variable('weight',shape,initializer=tf.contrib.layers.xavier_initializer())

def bias(shape,value=0.1):
	return tf.get_variable('bias',shape,initializer=tf.constant_initializer(value))

###########################################################
#define basic layers

def conv2D(x,size,outchn,name=None,stride=1,pad='SAME',activation=None,usebias=True,kernel_data=None,bias_data=None,dilation_rate=1):
	global l_num
	print('Conv_bias:',usebias)
	if name is None:
		name = 'conv_l_'+str(l_num)
		l_num+=1
	# with tf.variable_scope(name):
	if isinstance(size,list):
		kernel = size
	else:
		kernel = [size,size]
	if (not kernel_data is None) and (not bias_data is None):
		z = tf.layers.conv2d(x, outchn, kernel, strides=(stride, stride), padding=pad,\
			dilation_rate=dilation_rate,\
			kernel_initializer=tf.constant_initializer(kernel_data),\
			use_bias=usebias,\
			bias_initializer=tf.constant_initializer(bias_data),name=name)
	else:
		z = tf.layers.conv2d(x, outchn, kernel, strides=(stride, stride), padding=pad,\
			dilation_rate=dilation_rate,\
			kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),\
			use_bias=usebias,\
			bias_initializer=tf.constant_initializer(0.1),name=name)
	return z

def sum(x,y):
	return x+y

def deconv2D(x,size,outchn,name,stride=1,pad='SAME'):
	with tf.variable_scope(name):
		if isinstance(size,list):
			kernel = size
		else:
			kernel = [size,size]
		z = tf.layers.conv2d_transpose(x, outchn, [size, size], strides=(stride, stride), padding=pad,\
			kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),\
			bias_initializer=tf.constant_initializer(0.1))
		return z

def conv2Ddw(x,inshape,size,multi,name,stride=1,pad='SAME',weight_data=None):
	with tf.variable_scope(name):
		if isinstance(size,list):
			kernel = [size[0],size[1],inshape,multi]
		else:
			kernel = [size,size,inshape,multi]
		if weight_data==None:
			w = weight(kernel)
		else:
			w = weight_data
		res = tf.nn.depthwise_conv2d(x,w,[1,stride,stride,1],padding=pad)
	return res

def maxpooling(x,size,stride=None,name=None,pad='SAME'):
	global l_num
	if name is None:
		name = 'maxpooling_l_'+str(l_num)
		l_num+=1
	with tf.variable_scope(name):
		if stride is None:
			stride = size
		return tf.nn.max_pool(x,ksize=[1,size,size,1],strides=[1,stride,stride,1],padding=pad)

def avgpooling(x,size,stride=None,name=None,pad='SAME'):
	global l_num
	if name is None:
		name = 'avgpooling_l_'+str(l_num)
		l_num+=1
	with tf.variable_scope(name):
		if stride is None:
			stride = size
		return tf.nn.avg_pool(x,ksize=[1,size,size,1],strides=[1,stride,stride,1],padding=pad)

def Fcnn(x,insize,outsize,name,activation=None,nobias=False):
	with tf.variable_scope(name):
		if nobias:
			print('No biased fully connected layer is used!')
			W = weight([insize,outsize])
			tf.summary.histogram(name+'/weight',W)
			if activation==None:
				return tf.matmul(x,W)
			return activation(tf.matmul(x,W))
		else:
			W = weight([insize,outsize])
			b = bias([outsize])
			tf.summary.histogram(name+'/weight',W)
			tf.summary.histogram(name+'/bias',b)
			if activation==None:
				return tf.matmul(x,W)+b
			return activation(tf.matmul(x,W)+b)

def MFM(x,half,name):
	with tf.variable_scope(name):
		#shape is in format [batchsize, x, y, channel]
		# shape = tf.shape(x)
		shape = x.get_shape().as_list()
		res = tf.reshape(x,[-1,shape[1],shape[2],2,shape[-1]//2])
		res = tf.reduce_max(res,axis=[3])
		return res

def MFMfc(x,half,name):
	with tf.variable_scope(name):
		shape = x.get_shape().as_list()
		# print('fcshape:',shape)
		res = tf.reduce_max(tf.reshape(x,[-1,2,shape[-1]//2]),reduction_indices=[1])
	return res

def accuracy(pred,y,name):
	with tf.variable_scope(name):
		correct = tf.equal(tf.cast(tf.argmax(pred,1),tf.int64),tf.cast(y,tf.int64))
		acc = tf.reduce_mean(tf.cast(correct,tf.float32))
		#acc = tf.cast(correct,tf.float32)
		return acc

def batch_norm(inp,name,epsilon=None,variance=None,training=True):
	print('BN training:',training)
	if not epsilon is None:
		return tf.layers.batch_normalization(inp,training=training,name=name,epsilon=epsilon)
	return tf.layers.batch_normalization(inp,training=training,name=name)

def lrelu(x,name,leaky=0.2):
	return tf.maximum(x,x*leaky,name=name)

def relu(inp,name):
	return tf.nn.relu(inp,name=name)

def tanh(inp,name):
	return tf.tanh(inp,name=name)

def elu(inp,name):
	return tf.nn.elu(inp,name=name)

def sigmoid(inp,name):
	return tf.sigmoid(inp,name=name)

def resize_nn(inp,size,name):
	with tf.name_scope(name):
		if isinstance(size,list):
			return tf.image.resize_nearest_neighbor(inp,size=(int(size[0]),int(size[1])))
		elif isinstance(size,tf.Tensor):
			return tf.image.resize_nearest_neighbor(inp,size=size)
		else:
			return tf.image.resize_nearest_neighbor(inp,size=(int(size),int(size)))

def upSampling(inp,multiplier,name):
	b,h,w,c = inp.get_shape().as_list()
	if isinstance(multiplier,list):
		h2 = h*multiplier[0]
		w2 = w*multiplier[1]
	else:
		h2 = h*multiplier
		w2 = w*multiplier
	return resize_nn(inp,[h2,w2],name)
