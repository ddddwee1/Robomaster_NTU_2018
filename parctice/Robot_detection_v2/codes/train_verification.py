import tensorflow as tf
import graph

def train_verification(imgholder, croppedholder, veri_conf_holder, veri_conf_loss, verfication_train_step, conf, bias, veri_conf, feature_map)

	MAXITER = 50000*2
	BSIZE = 5
	with tf.Session() as sess:
		M.loadSess('./model/',sess,init=True)
		saver = tf.train.Saver()
		reader = data_reader()
		print('Reading finish')
		for iteration in range(MAXITER):
			# Getting the data from data_reader
			train_batch = reader.next_train_batch(BSIZE)
			img_batch = [i[0] for i in train_batch]
			bias_batch = [i[1] for i in train_batch]
			img_batch = np.float32(img_batch)
			
			# Running the RPN with the original images
			feeddict = {imgholder:img_batch}
			c, b, f, acc = sess.run([conf, bias, feature_map, accuracy], feed_dict=feeddict)
			
			# Cropped the feature maps and get the confidence label
			croppedFMs, labelConf = multipleFeatureMap(f, b, c, bias_batch, BSIZE)

			# Train the Verification Network
			feeddict_veri = {croppedholder:croppedFMs, veri_conf_holder:labelConf}
			sess.run([veri_conf_loss, verfication_train_step, veri_conf], feed_dict=feeddict_veri)
			
			if iteration%10==0
				print('Iter:',iteration,'\tLoss:',veri_conf_loss, '\tAccuracy : ', acc)		

				img = img_batch[0].astype(np.uint8)
				draw(img,c[0],b[0],wait=5)

	 
			if iteration%5000==0 and iteration!=0:
			saver.save(sess,'./model/'+str(iteration)+'.ckpt')

RPNholders, veriholders, RPNlosses, verilosses, train_steps, RPNcb, veri_confidence, feature_map = graph.build_graph()
train_RPN(RPNholders[0], veriholders[0], veriholders[1], verilosses[0], train_steps[1], RPNcb[0], RPNcb[1], veri_confidence, feature_map)
