import numpy as np 
from math import sqrt
import cv2 

# delete later
def normalize_uwb(w,h,ang):
	w = (w+5)/10.
	h = (h+5)/10.
	# change the bias while testing 
	ang_bias = -1-18-15-25-15 #************************************************************************
	w_offset = 0
	h_offset = -1
	# add post process here
	w_scale = 1.
	h_scale = 1.
	w = 80 - w 
	h = 50 - h 
	#print 'ang before',ang
	ang = -(ang+450)/900. + ang_bias
	ang = ang%40
	#print 'ang after',ang
	return (w+w_offset)*w_scale , (h+h_offset)*h_scale , ang


def get_delta(coord):
	if len(coord) == 0:
		return 0,0
	distance = [(x-240)**2+(y-320)**2 for x,y,w,h in coord]
	min_index = np.argmax(-np.array(distance))
	target_coord = coord[min_index]
	x,y,w,h = target_coord

	# do process on x,y,w,h
	return int(float(y-240)*3000/240),int(float(x-320)*3000/320)


