import rune_recog_template
import cv2
import time
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS,30)
#fps = cap.get(cv2.CAP_PROP_FPS)
#print 'fps',fps
cap.set(3, 800);
cap.set(4, 600);

while True:
	#img = cv2.imread('b.jpg',3)
	_,img = cap.read()
	#print img
	img1=np.array(img)
	#print img1.shape
	cv2.imshow('img',img)
	cv2.waitKey(1)
	try:
		handwritten_num , num_7seg = rune_recog_template.get_detection_rune(img)
	except:
		print 'Error'
		time.sleep(0.1)
		continue
	
	handwritten_dict = {}
	num_7seg_dict = {}

	# filter handwritten_num
	for i in range(len(handwritten_num)):
		handwritten_dict[handwritten_num[i]] = np.count_nonzero(handwritten_num[i])

	for i in range(len(num_7seg)):
		num_7seg_dict[num_7seg[i]] = np.count_nonzero(num_7seg[i])

	if len(handwritten_dict) != 9 and len(num_7seg_dict) != 5:
		print "wrong handwritten number"
	else:
		print handwritten_num
	
	if len(num_7seg_dict) != 5:
		print "wrong 7seg number"
	else:
		print num_7seg
	time.sleep(0.1)
