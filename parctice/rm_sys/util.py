import numpy as np 
from math import sqrt
import cv2 

last_shoot = 2
shoot_delay_conter = 0

def get_v_and_shoot(valid_coord,pid,img):
	global shoot_delay_conter
	#print img.shape
	if valid_coord != []:
		dist = [a[0]*a[0]+a[1]*a[1] for a in valid_coord]
		selected = np.argmax(-np.array(dist))
		width = valid_coord[selected][2]
		height = valid_coord[selected][3]
		x = valid_coord[selected][0]
		y = valid_coord[selected][1]
		x_bias = 2.1*sqrt(np.log(width * height)) + 1
		y_bias = 14*sqrt(np.log(width * height)) + 12
		
		# adjust with offset
		x_offset = x - img.shape[1]//2 - x_bias
		y_offset = y - img.shape[0]//2 - y_bias
		
		# Shooting logic
		if abs(x_offset) < 16+3.6*sqrt(np.log(width*height))/2 and abs(y_offset) < 12  :
			#pid.eval([x_offset, y_offset]) # must update pid to maintain the time
			#v_2 = 0
			#v_1 = 0
			v_2, v_1= pid.eval([x_offset, y_offset])
			if last_shoot==2:
				shoot = 2
			else:
				shoot = 2
			shoot_delay_conter = 7
		else:
			v_2, v_1= pid.eval([x_offset, y_offset])
			shoot = 0

		# draw centre of the camera
		#cv2.rectangle(img,(x-width//2,y-height//2),(x+width//2,y+height//2),(0,255,0),2)
		#cv2.rectangle(img,(img.shape[1]//2-width//2,img.shape[0]//2-height//2),(img.shape[1]//2+width//2,img.shape[0]//2+height//2),(0,0,255),2)

		#cv2.imshow('result',img)
		#cv2.waitKey(1)
		return v_1,v_2,shoot

	#cv2.imshow('result',img)
	#cv2.waitKey(1)
	
	# count down for shooting 
	if shoot_delay_conter >0:
		print 'delay counter',shoot_delay_conter
		shoot_delay_conter -= 1
		shoot = 2
	else:
		shoot = 0
	return 0,0,shoot

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

def encode_whang(w,h,ang):
	whang = np.int16([w,h,ang])
	encoded = whang.tobytes()
	return encoded

def decode_whang(whang):
	whang = np.frombuffer(whang,dtype=np.int16)
	w,h,ang = whang[0],whang[1],whang[2]
	return w,h,ang
