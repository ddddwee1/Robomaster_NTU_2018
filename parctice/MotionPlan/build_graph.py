import tensorflow as tf 
import model as M 
import numpy as np 
import net_parts as net

def build_graph1():
	env_holder = tf.placeholder(tf.float32,[None,4])
	Q_holder = tf.placeholder(tf.float32,[None,1])
	act_holder = tf.placeholder(tf.float32,[None,10])

	action = net.build_actor(env_holder,'actor1')
	gp0,gp1,gp2,gp3 = net.grouping(action)
	activated_action = net.activate_group([gp0,gp1,gp2,gp3])
	Q_est = net.build_crit(activated_action,env_holder,'crit1')

	crit_loss = tf.reduce_mean(tf.square(Q_est - Q_holder))

	lb0,lb1,lb2,lb3 = net.grouping(act_holder)

	loss_a_0 = tf.nn.softmax_cross_entropy_with_logits(logits=gp0,labels=lb0)
	loss_a_1 = tf.nn.softmax_cross_entropy_with_logits(logits=gp1,labels=lb1)
	loss_a_2 = tf.nn.softmax_cross_entropy_with_logits(logits=gp2,labels=lb2)
	loss_a_3 = tf.nn.sigmoid_cross_entropy_with_logits(logits=gp3,labels=lb3)

	act_loss = loss_a_0+loss_a_1+loss_a_2+loss_a_3

	var_act = M.get_trainable_vars('actor1')
	var_crit = M.get_trainable_vars('crit1')

	train_crit_human = tf.train.AdamOptimizer(0.0001).minimize(crit_loss,var_list=var_crit)
	train_act_human = tf.train.AdamOptimizer(0.0001).minimize(act_loss,var_list=var_act)
	
	holders = [env_holder, Q_holder, act_holder]
	losses = [act_loss,crit_loss]
	train_steps = [train_act_human, train_crit_human]

	return holders, losses, train_steps


def build_graph2():
	env_holder = tf.placeholder(tf.float32,[None,4])
	Q_holder = tf.placeholder(tf.float32,[None,1])

	action = net.build_actor(env_holder,'actor1')
	gps = net.grouping(action)
	activated_gps = net.activate_group(gps)
	Q_estimated = net.build_crit(activated_gps,env_holder,'cirt1')
	Q_real = net.build_crit(activated_gps,env_holder,'crit2')

	act_loss = -Q_real
	crit_loss = tf.reduce_mean( tf.square(Q_estimated - Q_holder))

	var_act = M.get_trainable_vars('actor1')
	var_est = M.get_trainable_vars('crit1')
	var_real = M.get_trainable_vars('crit2')

	train_act = tf.train.AdamOptimizer(0.0001).minimize(act_loss,var_list=var_act)
	train_crit = tf.train.AdamOptimizer(0.0001).minimize(crit_loss,var_list=var_est)

	holders = [env_holder,Q_holder]
	VARS = [var_act,var_est,var_real]
	losses = [act_loss, crit_loss]
	train_steps = [train_act,train_crit]

	return holders, losses, train_steps, Q_real, activated_gps, VARS