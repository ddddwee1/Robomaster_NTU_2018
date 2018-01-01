import gamemain
import net
import tensorflow as tf
import numpy as np 

# gamemain.get_next_frame([0,0,0,0,0,0,0,1])

# gamemain.get_next_frame([0,0,0,0,0,0,0,0])

# gamemain.get_next_frame([0,0,0,0,0,0,0,0])

# gamemain.get_next_frame([0,0,0,0,0,0,0,1])

# gamemain.get_next_frame([0,0,0,0,0,0,0,0])

# gamemain.get_next_frame([0,0,0,0,0,0,0,1])

class population():
	def __init__(self,sess,inpholder,vars,net):
		self.pop = []
		for i in range(14):
			self.sess = sess
			self.inpholder = inpholder
			sess.run(tf.global_variables_initializer())
			a,b,c,d = sess.run(vars)
			a = [a,b,c,d]
			self.pop.append(a)
			self.net = net
			self.vars = vars

	def next_gen(self,rank):
		top2 = []
		for i in range(len(rank)):
			if rank[i]==0 or rank[i]==1:
				top3.append(self.pop[i])

		# Generate next generation
		nextpop = []
		nextpop += top2 
		nextpop.append(net.mutate_all(top2[0],0.01))
		nextpop.append(net.mutate_all(top2[1],0.01))
		nextpop.append(net.cross_all(top2[0],top2[1],0.5))
		nextpop.append(net.mutate_all(nextpop[-1],0.01))
		nextpop.append(net.cross_all(top2[0],top2[1],0.25))
		nextpop.append(net.mutate_all(nextpop[-1],0.01))
		nextpop.append(net.cross_all(top2[0],top2[1],0.75))
		nextpop.append(net.mutate_all(nextpop[-1],0.01))
		for _ in range(4):
			buf = random.sample(self.pop,1)[0]
			nextpop.append(net.mutate_all(buf,0.01))
		self.pop = nextpop

	def assign(self,test_num):
		assign_tensor = [tf.assign(i,j) for i,j in zip(self.vars,self.pop[test_num])]
		self.sess.run(assign_tensor)

	def test_one(self,test_num):
		self.assign(test_num)
		do_nothing = np.zeros([1,4])
		do_nothing[0][3] = 1
		gamemain.reset()
		ang, dis,reward = gamemain.get_next_frame(do_nothing[0])
		reward_ttl = 0
		for _ in range(1000):
			act = sess.run(self.net,feed_dict={self,inpholder:[[ang,dis]]})
			ang, dis, reward = gamemain.get_next_frame(act[0])
			reward_ttl += reward
		return reward_ttl

	def test_all(self):
		a = []
		for i in range(len(self.pop)):
			reward = self.test_one(i)
			a.append(reward)
		a = np.array(a)
		a_arg = np.argsort(-a)
		rank = np.zeros([1,len(a)])
		for i in range(len(a_arg)):
			rank[a_arg[i]] = i
		return rank,a.max()


inpholder, var, output = net.turret()
with tf.Session() as sess:
	pop = population(sess,inpholder,var,output)
	for i in range(100):
		rank,fit_max = pop.test_all()
		pop.next_gen(rank)
		print('Gen%d:\t%.1f'%(i,fit_max))
