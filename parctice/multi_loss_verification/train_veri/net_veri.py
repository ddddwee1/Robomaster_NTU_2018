import tensorflow as tf 
import model as M 

def build_model(inp):
	with tf.variable_scope('VERI'):
		mod = M.Model(inp)
		mod.convLayer(5,16,stride=2,activation=M.PARAM_LRELU)
		mod.convLayer(5,32,stride=2,activation=M.PARAM_LRELU)
		mod.convLayer(3,32,stride=2,activation=M.PARAM_LRELU) #4
		mod.flatten()
		mod.fcLayer(2)
	return mod.get_current_layer()

def build_loss(layer,lb):
	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=layer,labels=lb))
	ts = tf.train.AdamOptimizer(0.0001).minimize(loss)
	return loss,ts

inputholder = tf.placeholder(tf.float32,[None,32,32,3])
labelholder = tf.placeholder(tf.int32,[None])

output = build_model(inputholder)
accuracy = M.accuracy(output,labelholder)
loss,ts = build_loss(output,labelholder)