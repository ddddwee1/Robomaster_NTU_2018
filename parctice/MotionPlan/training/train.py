import tensorflow as tf 
import model as M 
import numpy as np 
import build_graph as graph 
from data_reader import data_reader

def assign_vars(var1,var2):
	for i,j in zip(var1,var2):
		tf.assign(i,j)

def train_human():
	BSIZE = 128
	MAXITER = 100000
	holders, losses, trainsteps = graph.build_graph1()
	saver = tf.train.Saver()
	with tf.Session() as sess:
		M.loadSess('./model/',sess,init=True)
		reader = data_reader()
		for i in range(MAXITER):
			train_batch = reader.next_batch(BSIZE)
			env_batch = [i[0] for i in train_batch]
			Q_batch = [i[1] for i in train_batch]
			act_batch = [i[2] for i in train_batch]
			feed_d = {holders[0]:env_batch, holders[1]:Q_batch, holders[2]:act_batch}
			act_loss, crit_loss, _, _ = sess.run(losses+trainsteps, feed_dict=feed_d)
			if i%100==0:
				print('Iter:%d\tAct_loss:%.4f\tCrit_loss:%.4f\t'%(i,act_loss,crit_loss))
			if i%100==0:
				train_batch = reader.next_batch(BSIZE)
				env_batch = [i[0] for i in train_batch]
				Q_batch = [i[1] for i in train_batch]
				act_batch = [i[2] for i in train_batch]
				feed_d = {holders[0]:env_batch, holders[1]:Q_batch, holders[2]:act_batch}
				act_loss, crit_loss = sess.run(losses, feed_dict=feed_d)
				print('Valid\tAct_loss:%.4f\tCrit_loss:%.4f\t'%(act_loss,crit_loss))
			if i%5000==0 and i>0:
				saver.save(sess,'./model/%d.ckpt'%(i))

def train_self():
	BSIZE = 128
	MEM_SIZE = 30000
	holders, losses, train_steps, Q_real, activated_gps, VARS = graph.build_graph2()
	saver = tf.train.Saver(var_list=VARS[0]+VARS[1])
	with tf.Session() as sess:
		M.loadSess('./model/',sess,var_list=VARS[0]+VARS[1])
		assign_vars(VARS[2],VARS[1])
		# while True:
			#Game operations
			
train_human()