import netpart
import net_veri
import tensorflow as tf 
import model as M 

with tf.Session() as sess:
	M.loadSess('./modelveri/',sess)
	saver = tf.train.Saver()
	saver.save(sess,'./modelveri_tiny/MSRPN_triple.ckpt')