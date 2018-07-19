# Remove training info.

import model as M 
import tensorflow as tf 

import net_veri
import netpart

import os 

if not os.path.exists('./modelveri_tiny/'):
	os.mkdir('./modelveri_tiny/')

with tf.Session() as sess:
	M.loadSess('./modelveri/',sess)
	saver = tf.train.Saver()
	saver.save(sess,'./modelveri_tiny/armour_plate.ckpt')