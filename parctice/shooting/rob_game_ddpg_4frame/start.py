import numpy as np 
import tensorflow as tf 
import net 
import memory
import gamemain
import model as M
import random

#initialize game
gamemain.reset()

#initialize network
holders, action, losses, train_steps, assign, init_assign = net.build_graph()

# Constants
EXPLORE = 10000
TRAINING = 200000
INIT_RAND = 0.7
FINAL_RAND = 0.0
rand_rate = 0.7
BSIZE = 256
# Explore and training
frame_count = 0
var = 3
episode = 0

sess = tf.Session()
saver = tf.train.Saver()
M.loadSess('./model/',sess,init=True)
sess.run(init_assign)

# init frame
do_nothing = np.float32([0,-0.8])
r,rad,reward,terminated = gamemain.get_next_frame(do_nothing)
env0 = [r,rad]
# act = sess.run(action,feed_dict={holders[0]:[env0]})
# r,rad,reward,terminated = gamemain.get_next_frame(act[0])
# env1 = [r,rad]
# memory.push([env0,env1,act[0],[reward],[terminated]])
# env0 = env1
env0 = env0 * 4

while True:
	act = sess.run(action,feed_dict={holders[0]:[env0]})
	act = act[0]
	act = np.random.normal(act,var)
	act = np.clip(act,-1.,1.)
	r,rad,reward,terminated = gamemain.get_next_frame(act)
	env1 = env0[2:] + [r,rad]
	memory.push([env0,env1,act,[reward],[terminated]])
	if reward>7:
		memory.push_prior([env0,env1,act,[reward],[terminated]])
	if terminated==0:
		gamemain.reset()
		r,rad,reward,terminated = gamemain.get_next_frame(do_nothing)
		env0 = [r,rad]
		env0 = env0 * 4
		var = var*0.95
		episode += 1
	else:
		env0 = env1
	frame_count += 1

	# training
	if episode>=1:
		train_batch = memory.next_batch(BSIZE)
		s0_batch = [i[0] for i in train_batch]
		# print(s0_batch[0])
		s1_batch = [i[1] for i in train_batch]
		a_batch = [i[2] for i in train_batch]
		rw_batch = [i[3] for i in train_batch]
		t_batch = [i[4] for i in train_batch]
		feed_d = {holders[0]:s0_batch,holders[1]:s1_batch,holders[2]:rw_batch,holders[3]:a_batch,holders[4]:t_batch}
		c_loss, a_loss, _,_ = sess.run(losses+train_steps,feed_dict=feed_d)
		# c_loss, a_loss = sess.run(losses,feed_dict=feed_d)
		if frame_count%100==0:
			sess.run(assign)

		if frame_count%5000==0:
			saver.save(sess,'./model/%d.ckpt'%(frame_count))
		if frame_count%100==0:
			print('Frame:%d\tC_Loss:%.4f\tA_Loss:%.4f\tEpsilon:%.4f'%(frame_count,c_loss,a_loss,var))