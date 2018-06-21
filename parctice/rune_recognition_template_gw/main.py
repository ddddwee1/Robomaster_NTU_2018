import rune_recog_template
import cv2
import time
import numpy as np
#import data_retriver
#import robot_prop 
#import util
#import detection_mod
#from camera_module import camera_thread

#data_reader = data_retriver.data_reader_thread()
#data_reader.start()

#camera_thread = camera_thread()
#camera_thread.start()

turret_pitch = [ 600 , 200 , -200] 
turret_yaw = [ -500 , 0 , -500]


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS,10)
fps = cap.get(cv2.CAP_PROP_FPS)
print 'fps',fps
cap.set(3, 640);
cap.set(4, 480);

robot_prop.v1 = 0
robot_prop.v2 = 0
save_num_7seg = 0
num_7seg_index = 0

while True:
	#img = cv2.imread('b.jpg',3)
	_,img = cap.read()
	#print img
	try:
		handwritten_num_raw , num_7seg_raw = rune_recog_template.get_detection_rune(img)
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
		handwritten_num= handwritten_num_raw
		print handwritten_num
	
	if len(num_7seg_dict) != 5:
		print "wrong 7seg number"
	else:
		num_7seg = num_7seg_raw
		print num_7seg

	if save_num_7seg == 0 or save_num_7seg == num_7seg and handwritten_num != None and num_7seg!=None :
		shoot_handwritten_num = num_7seg[num_7seg_index]
		num_7seg_index +=1
		for i in range(len(handwritten_num)):
			if handwritten_num[i] == shoot_handwritten_num:
				handwritten_num_index = i
				continue

		handwritten_num_row = handwritten_num // 3
		handwritten_num_col = handwritten_num % 3

		robot_prop.v1 = turret_pitch[int(handwritten_num_row)]
		robot_prop.v2 = turret_yaw[int(handwritten_num_col)]
		for i in range(20):
			robot_prop.shoot = 1
		time.sleep(0.05)
		robot_prop.v1 = 0
		robot_prop.v2 = 0
		robot_prop.shoot = 0

	else:
		save_num_7seg = 0
		num_7seg_index = 0

	save_num_7seg = num_7seg
	#time.sleep(0.2)
	cv2.imshow('img',img)
	cv2.waitKey(1)
