import numpy as np 
import netpart
import data_reader
import model as M
import tensorflow as tf 
import cv2 

reader = data_reader.reader(height=540,width=960,scale_range=[0.05,0.5],
	lower_bound=3,upper_bound=5,index_multiplier=2)

def draw(img,c,b,multip,name):
	c = c[0]
	b = b[0]
	row,col,_ = b.shape
	# print(b.shape,c.shape)
	# print(row,col)
	for i in range(row):
		for j in range(col):
			# print(i,j)
			if c[i][j][0]>-0.5:
				x = int(b[i][j][0])+j*multip+multip//2
				y = int(b[i][j][1])+i*multip+multip//2
				w = int(b[i][j][2])
				h = int(b[i][j][3])
				cv2.rectangle(img,(x-w//2,y-h//2),(x+w//2,y+h//2),(0,255,0),2)
	cv2.imshow(name,img)
	cv2.waitKey(1)

b0,b1,c0,c1 = netpart.model_out
netout = [[b0,c0],[b1,c1]]

MAX_ITER = 100000
with tf.Session() as sess:
	saver = tf.train.Saver()
	M.loadSess('./model/',sess)
	for i in range(MAX_ITER):
		img, train_dic = reader.get_img()
		for k in train_dic:
			ls,_,b,c = sess.run([netpart.loss_functions[k],
				netpart.train_steps[k]] + netout[k],
				feed_dict={netpart.inpholder:[img],
				netpart.b_labholder:[train_dic[k][1]],
				netpart.c_labholder:[train_dic[k][0]]})
		if i%10==0:
			print('Iter:\t%d\tLoss:\t%.6f\n'%(i,ls))
		if i%100==0:
			multip = 8 if k==0 else 32
			draw(img.copy(),[train_dic[k][0]],[train_dic[k][1]],multip,'lab')
			draw(img.copy(),c,b,multip,'pred')
		if i%2000==0 and i>0:
			saver.save(sess,'./model/MSRPN_%d.ckpt'%i)