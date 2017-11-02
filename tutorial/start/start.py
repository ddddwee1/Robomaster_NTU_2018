import tensorflow as tf 
import numpy as np 
import model as M 

def build_graph():
	a = tf.constant(3) # tf.Variable(3)
	b = tf.constant(4)

	c = a+b 
	return c

c = build_graph()
with tf.Session() as sess:
	# M.loadSess('./model/',sess,init=True)
	print(sess.run(c))