import tensorflow as tf 
import model as M 
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def build_model(img_input):
	mod = M.Model(img_input,[None,28*28])
	mod.fcLayer(300,activation=M.PARAM_SIGMOID)
	mod.fcLayer(200,activation=M.PARAM_SIGMOID)
	mod.fcLayer(10)
	return mod.get_current_layer()

def build_graph():
	img_holder = tf.placeholder(tf.float32,[None,28*28])
	lab_holder = tf.placeholder(tf.float32,[None,10])
	last_layer = build_model(img_holder)
	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=lab_holder,logits=last_layer))
	accuracy = M.accuracy(last_layer,tf.argmax(lab_holder,1))
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
	return img_holder,lab_holder,loss,train_step,accuracy

img_holder,lab_holder,loss,train_step,accuracy = build_graph()
MAX_ITER = 50000
BSIZE = 128

with tf.Session() as sess:
	saver = tf.train.Saver()
	M.loadSess('./model/',sess,init=True)
	for i in range(50000):
		x_train, y_train = mnist.train.next_batch(BSIZE)
		_,acc,ls = sess.run([train_step,accuracy,loss],feed_dict={img_holder:x_train,lab_holder:y_train})
		if i%100==0:
			print('iter',i,'\t|acc:',acc,'\tloss:',ls)
		if i%1000==0:
			acc = sess.run(accuracy,feed_dict={img_holder:mnist.test.images, lab_holder:mnist.test.labels})
			print('Validation:',acc)
	saver.save(sess,'./model/abc.ckpt')