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

inputholder = tf.placeholder(tf.float32,[None,32,32,3])

output = build_model(inputholder)
output = tf.nn.softmax(output,-1)