import numpy as np 
import netpart
import model as M
import tensorflow as tf 
import time
from tensorflow.python.client import timeline

netout = netpart.model_out

config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
run_metadata = tf.RunMetadata()

with tf.Session(config=config) as sess:
	M.loadSess('./model/',sess,init=True)
	print('start')
	
	img = np.random.random([1,480,640,3])
	t1 = time.time()
	for i in range(1000):
		print(i)
		
		# b = sess.run(netout,options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),run_metadata=run_metadata)
		# b = sess.run(netout,feed_dict={netpart.inpholder:img},options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),run_metadata=run_metadata)
		b = sess.run(netout,feed_dict={netpart.inpholder:img})

		
		# trace = timeline.Timeline(step_stats=run_metadata.step_stats)
		# with open('timeline.ctf.json', 'w') as trace_file:
		# 	trace_file.write(trace.generate_chrome_trace_format())
	t2 = time.time()
	print((t2-t1))
	