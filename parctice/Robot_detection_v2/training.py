import tensorflow as tf



def draw(img,c,b,wait=0):
	# print(c.shape)
	# print(b.shape)
	# print(c.max())
	for i in range(16):
		for j in range(16):
			if c[i][j][0]>0:
				x = int(b[i][j][0])+j*16+8
				y = int(b[i][j][1])+i*16+8
				w = int(b[i][j][2])
				h = int(b[i][j][3])
				# print(b[i][j])
				# cv2.circle(img,(x,y),5,(0,0,255),-1)
				cv2.rectangle(img,(x-w,y-h),(x+w,y+h),(0,255,0),2)
	cv2.imshow('img',img)
	cv2.waitKey(wait)
	# cv2.destroyAllWindows()

def train_RBN(imgholder,biasholder,confholder,bias_loss,conf_loss,train_step,conf,bias)

	MAXITER = 50000*2
	BSIZE = 32
	with tf.Session() as sess:
		M.loadSess('./model/',sess,init=True)
		saver = tf.train.Saver()
		reader = data_reader()
		print('Reading finish')
		for iteration in range(MAXITER):
			train_batch = reader.next_train_batch(BSIZE)
			img_batch = [i[0] for i in train_batch]
			bias_batch = [i[1] for i in train_batch]
			conf_batch = [i[2] for i in train_batch]
			img_batch = np.float32(img_batch)
			feeddict = {imgholder:img_batch, biasholder:bias_batch, confholder:conf_batch}
			b_loss, c_loss, _, c, b	= sess.run([bias_loss,conf_loss,train_step,conf, bias],feed_dict=feeddict)
			if iteration%10==0
				print('Iter:',iteration,'\tLoss_b:',b_loss,'\tLoss_c:',c_loss)		
				# print(c.max())
				img = img_batch[0].astype(np.uint8)
				draw(img,c[0],b[0],wait=5)
				# draw(img_batch[0],conf_batch[0],bias_batch[0],wait=5)
	 
			if iteration%5000==0 and iteration!=0:
				saver.save(sess,'./model/'+str(iteration)+'.ckpt')

def train_veri(feature,c,b,imgholder,biasholder,confholder,cropholder,conf,bias,veri_biasholder,veri_confholder,veri_conf_loss,veri_bias_loss,veri_train_step)

	MAXITER = 50000*2
	BSIZE = 32
	with tf.Session() as sess:
		M.loadSess('./model/',sess,init=false)
		saver = tf.train.Saver()
		for iteration in range(MAXITER):
			train_batch = reader.next_train_batch(BSIZE)
			img_batch = [i[0] for i in train_batch]
			bias_batch = [i[1] for i in train_batch]
			conf_batch = [i[2] for i in train_batch]
			img_batch = np.float32(img_batch)
			feeddict = {imgholder:img_batch, biasholder:bias_batch, confholder:conf_batch}
			feature, b, c = sess.run([feature,bias,conf],feed_dict=feeddict)
			crop_batch,v_bias_batch,v_conf_batch = cropfunction(feature,c,b)
			feeddict_veri = {cropholder:crop_batch, veri_biasholder:v_bias_batch, veri_confholder:v_conf_batch}
			c_loss, _, _= sess.run([veri_conf_loss, veri_bias_loss, veri_train_step],feed_dict=feeddict_veri)
			if iteration%10==0
				print('Iter:',iteration,'\tConf_loss:',c_loss)
			if itration%5000==0 and iteration!=0:
				saver.save(sess,'./model/'+str(iteration)+'.ckpt')