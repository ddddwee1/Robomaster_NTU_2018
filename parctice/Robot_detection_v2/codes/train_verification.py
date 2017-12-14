import tensorflow as tf
import graph

def train_verification(imgholder,verfication_train_step,veri_conf,feature_map)

	MAXITER = 50000*2
	BSIZE = 1
	with tf.Session() as sess:
		M.loadSess('./model/',sess,init=True)
		saver = tf.train.Saver()
		reader = data_reader()
		print('Reading finish')
		for iteration in range(MAXITER):
			train_batch = reader.next_train_batch(BSIZE)
			img_batch = [i[0] for i in train_batch]
			bias_batch = [i[1] for i in train_batch]
			img_batch = np.float32(img_batch)
			feeddict = {imgholder:img_batch}
			_, c, b, f = sess.run([verfication_train_step,conf, bias, feature_map],feed_dict=feeddict)
			if iteration%10==0
				print('Iter:',iteration,'\tLoss_b:',b_loss,'\tLoss_c:',c_loss)		
				# print(c.max())
				img = img_batch[0].astype(np.uint8)
				draw(img,c[0],b[0],wait=5)
				# draw(img_batch[0],conf_batch[0],bias_batch[0],wait=5)
	 
			if iteration%5000==0 and iteration!=0:
		saver.save(sess,'./model/'+str(iteration)+'.ckpt')

RPNholders, veriholders, RPNlosses, verilosses, train_steps, RPNcb, veri_confidence, feature_map = graph.build_graph()
train_RPN(RPNholders[0],train_steps[1], veri_confidence, feature_map)