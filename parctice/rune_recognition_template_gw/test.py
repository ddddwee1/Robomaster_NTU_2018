import rune_recog_template
import cv2
import time
import numpy as np

RESIZING_PROCESSED_IMAGE = True

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS,10)
fps = cap.get(cv2.CAP_PROP_FPS)
print 'fps',fps
cap.set(3, 640);
cap.set(4, 480);

while True:
	#img = cv2.imread('b.jpg',3)
	_,img = cap.read()
	#print img
#	try:
#		handwritten_num = rune_recog_template.get_detection_rune(img)
#	except:
#		print 'Error'
#		time.sleep(0.1)
#		continue
	#print handwritten_num
#	handwritten_dict = {}
#	for i in range(len(handwritten_num)):
#		handwritten_dict[handwritten_num[i]] = np.count_nonzero(handwritten_num[i])
#		
#	if len(handwritten_dict) == 9:
#		print handwritten_num

#	time.sleep(0.2)
	cv2.imshow('img',img)
	cv2.waitKey(1)
