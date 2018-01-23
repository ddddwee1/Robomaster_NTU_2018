import tensorflow as tf 
import numpy as np 
import model as M 


EPS = 0.9

def actor(inp,scope):
	with tf.variable_scope(scope):
		mod = M.Model(inp,[None,8])
		mod.fcLayer(200,activation=M.PARAM_LRELU)
		mod.fcLayer(80,activation=M.PARAM_LRELU)
		mod.fcLayer(2,activation=M.PARAM_TANH)
	return mod.get_current_layer()

def crit(inp,scope,reuse=False):
	with tf.variable_scope(scope,reuse=reuse):
		mod = M.Model(inp,[None,10])
		mod.fcLayer(100,activation=M.PARAM_LRELU)
		mod.fcLayer(50,activation=M.PARAM_LRELU)
		mod.fcLayer(1)
	return mod.get_current_layer()

def build_graph():
	envholder = tf.placeholder(tf.float32,[None,8])
	envholder2 = tf.placeholder(tf.float32,[None,8])
	reward_holder = tf.placeholder(tf.float32,[None,1])
	aholder = tf.placeholder(tf.float32,[None,2])
	terminated_holder = tf.placeholder(tf.float32,[None,1])

	a_eval = actor(envholder,'a1')
	a_real = actor(envholder2,'a2')

	env_act1 = tf.concat([envholder,a_eval],axis=-1)
	env_act2 = tf.concat([envholder2,a_real],axis=-1)
	env_act3 = tf.concat([envholder,aholder],axis=-1)

	c_eval = crit(env_act1,'c1')
	c_real = crit(env_act2,'c2')
	c_real2 = crit(env_act1,'c2',True)
	c_eval2 = crit(env_act3,'c1',True)

	var_a1 = M.get_trainable_vars('a1')
	var_a2 = M.get_trainable_vars('a2')
	var_c1 = M.get_trainable_vars('c1')
	var_c2 = M.get_trainable_vars('c2')

	q_target = EPS*c_real*terminated_holder + reward_holder

	c_loss = tf.reduce_mean(tf.square(c_eval2 - q_target))

	a_loss = -tf.reduce_mean(c_real2)

	train_c = tf.train.RMSPropOptimizer(0.0002).minimize(c_loss,var_list=var_c1)
	train_a = tf.train.RMSPropOptimizer(0.0001).minimize(a_loss,var_list=var_a1)

	assign_a = soft_assign(var_a2,var_a1,0.5)
	assign_c = soft_assign(var_c2,var_c1,0.5)

	assign_a0 = assign(var_a2,var_a1)
	assign_c0 = assign(var_c2,var_c1)

	return [envholder,envholder2,reward_holder,aholder,terminated_holder],a_eval,[c_loss,a_loss],[train_c,train_a],[assign_c,assign_a],[assign_c0,assign_a0]

def soft_assign(old,new,tau):
	assign_op = [tf.assign(i,tau*j+(1-tau)*i) for i,j in zip(old,new)]
	return assign_op

def assign(old,new):
	assign_op = [tf.assign(i,j) for i,j in zip(old,new)]
	return assign_op