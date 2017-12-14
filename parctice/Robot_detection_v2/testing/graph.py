import tensorflow as tf 
from net_parts import verify_net, RPN

def build_graph(test=False):
	with tf.name_scope('imgholder'): # The placeholder is just a holder and doesn't contains the actual data.
		imgholder = tf.placeholder(tf.float32,[None,256,256,3]) # The 3 is color channels
	with tf.name_scope('bias_holder'):
		bias_holder = tf.placeholder(tf.float32,[None,16,16,4]) # The bias (x,y,w,h) for 16*16 feature maps.
	with tf.name_scope('conf_holder'):
		conf_holder = tf.placeholder(tf.float32,[None,16,16,1]) # The confidence about 16*16 feature maps.
	with tf.name_scope('croppedholder'):
		croppedholder = tf.placeholder(tf.float32,[None,16,16,256]) # 256 is the number of feature maps
	with tf.name_scope('veri_conf_holder'):
		veri_conf_holder = tf.placeholder(tf.float32, [None,1])
	with tf.name_scope('veri_bias_holder'):
		veri_bias_holder = tf.placeholder(tf.float32, [None,4]) # The veri output numbers,x,y,w,h

	with tf.name_scope('mask'):
		maskholder = tf.placeholder(tf.float32,[None,16,16,1])

	conf, bias,feature_map = RPN(imgholder,test=test)
	veri_conf, veri_bias = verify_net(croppedholder)
	
	bias_loss = tf.reduce_sum(tf.reduce_mean(tf.square(bias*conf_holder - bias_holder),axis=0))
	conf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=conf,labels=conf_holder))

	veri_bias_loss = tf.reduce_sum(tf.reduce_mean(tf.square(veri_bias*veri_conf_holder - veri_bias_holder),axis=0))
	veri_conf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=veri_conf,labels=veri_conf_holder))
	
	train_step = tf.train.AdamOptimizer(0.0001).minimize(bias_loss+conf_loss)
	veri_train_step = tf.train.AdamOptimizer(0.0001).minimize(veri_bias_loss+veri_conf_loss)

	return [imgholder,bias_holder, conf_holder,maskholder], [croppedholder, veri_bias_holder, veri_conf_holder], [bias_loss, conf_loss], [veri_bias_loss, veri_conf_loss], [train_step, veri_train_step], [conf, bias] , [veri_conf,veri_bias], feature_map
