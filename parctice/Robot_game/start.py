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
	def __init__(self,sess,var):
		self.pop = []
		for i in range(10):
			sess.run(tf.global_variables_initializer())
			a = sess.run(var)
			self.pop.append(a)