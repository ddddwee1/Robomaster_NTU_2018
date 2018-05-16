import tensorflow as tf 
import model as M
from net_parts import verify_net, multiloss_RPN

class build_graph:

	def __init__(self):
		self.data = []
		self.biaslosses = []
		self.conflosses = []


	def loss_function(self,bias,conf,b_lab,c_lab,coef=1.):
		bias_loss = tf.reduce_sum(tf.reduce_mean(tf.square(bias - b_lab)*c_lab,axis=0))
		conf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=c_lab,logits=conf))
		self.biaslosses.append(bias_loss)
		self.conflosses.append(conf_loss)
		return (bias_loss  + conf_loss)*coef

	def build_loss(self,b0,b1,b2,c0,c1,c2,b_lab,c_lab):
		loss_functions = []
		loss_functions.append(self.loss_function(b0,c0,b_lab,c_lab,2.8))
		loss_functions.append(self.loss_function(b1,c1,b_lab,c_lab))
		loss_functions.append(self.loss_function(b2,c2,b_lab,c_lab,1./2.8))
		return loss_functions

	def get_train_steps(self,losses):
		train_steps = []

		for loss in losses:
			step = tf.train.AdamOptimizer(0.000001).minimize(loss)
			train_steps.append(step)

		return train_steps

	def graphs(self, test=True):

		with tf.name_scope('imgholder'): # The placeholder is just a holder and doesn't contains the actual data.
			imgholder = tf.placeholder(tf.float32,[None,None,None,3]) # The 3 is color channels
		with tf.name_scope('bias_holder'):
			bias_holder = tf.placeholder(tf.float32,[None,None,None,4]) # The bias (x,y,w,h) for 16*16 feature maps.
		with tf.name_scope('conf_holder'):
			conf_holder = tf.placeholder(tf.float32,[None,None,None,1]) # The confidence about 16*16 feature maps.
		with tf.name_scope('croppedholder'):
			croppedholder = tf.placeholder(tf.float32,[None,24,24,3]) # 256 is the number of feature maps
		with tf.name_scope('veri_conf_holder2'):
			veri_conf_holder = tf.placeholder(tf.float32, [None,1])
#		with tf.name_scope('veri_bias_holder'):
#			veri_bias_holder = tf.placeholder(tf.float32, [None,4]) # The veri output numbers,x,y,w,h

		#with tf.name_scope('mask'):
		#	maskholder = tf.placeholder(tf.float32,[None,None,None,1])

		b0,b1,b2,c0,c1,c2 = multiloss_RPN(imgholder, test)
		loss_functions = self.build_loss(b0,b1,b2,c0,c1,c2,bias_holder,conf_holder)
		
		train_s = self.get_train_steps(loss_functions)


		veri_conf = verify_net(croppedholder, test)
#		veri_bias_loss = tf.reduce_sum(tf.reduce_mean(tf.square(veri_bias*veri_conf_holder - veri_bias_holder),axis=0))
		veri_conf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=veri_conf_holder,logits=veri_conf))

		veri_train_step = tf.train.AdamOptimizer(0.0001).minimize(veri_conf_loss)

		#veri_accuracy = M.accuracy(veri_conf, tf.argmax(veri_conf_holder,1))
		correct_pred = tf.equal(tf.round(tf.sigmoid(veri_conf)), tf.round(veri_conf_holder))
		veri_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

		return [imgholder,bias_holder, conf_holder], [croppedholder, veri_conf_holder], [self.biaslosses,self.conflosses] , [veri_conf_loss], [train_s, veri_train_step],[b0,b1,b2,c0,c1,c2], [veri_conf, veri_accuracy]

