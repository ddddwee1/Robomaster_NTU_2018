from net import model as M 
from net import netpart, net_veri
import tensorflow as tf 
import numpy as np
import cv2

def pre_process(img):
	img2 = cv2.blur(img,(5,5))

	hsv = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)

	# for red
	h1,h2,s1,s2,v1,v2 = 0,55,0,255,245,255

	# for blue
	# h1,h2,s1,s2,v1,v2 = 100,150,0,255,245,255

	# print(h1,h2)
	lower = np.array([h1,s1,v1])
	upper = np.array([h2,s2,v2])
	buf = cv2.inRange(hsv,lower,upper)

	empt = np.zeros(buf.shape)

	candidates = []
	_,contours, hierarchy = cv2.findContours(buf,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	for ctr in contours:
		cnt = cv2.boundingRect(ctr)
		if float(cnt[3])/float(cnt[2])>1.3:
			cv2.rectangle(empt,(cnt[0],cnt[1]),(cnt[0]+cnt[2],cnt[1]+cnt[3]),255,1)
			candidates.append(cnt)

	areas = []
	# region proposal
	for i in range(len(candidates)-1):
		for j in range(i+1,len(candidates)):
			cnt1 = candidates[i]
			cnt2 = candidates[j]
			# pair should be overlapped in y
			if cnt1[1]<=cnt2[1]+cnt2[3] and cnt2[1]<=cnt1[1]+cnt1[3]:
				max_h = max(cnt1[3],cnt2[3])
				# horizontal distance must less than 3 times max_h
				if abs(cnt1[0]-cnt2[0]) < 3*max_h:
					# propose this area
					if cnt1[0]<cnt2[0]:
						left_bar = cnt1
						right_bar = cnt2
					else:
						left_bar = cnt2
						right_bar = cnt1
					top_left = [left_bar[0]-left_bar[2]//2, left_bar[1] - max_h]
					btm_right = [right_bar[0]+right_bar[2]+right_bar[2]//2, right_bar[1]+right_bar[3]+max_h]
					areas.append([top_left,btm_right])
	return areas

def crop(areas,img):
	img_coord = []
	for top_left,btm_right in areas:
		w = btm_right[0] - top_left[0]
		h = btm_right[1] - top_left[1]
		tl0 = top_left[0] - int(0.25*w)
		tl1 = top_left[1] - int(0.25*h)
		br0 = btm_right[0] + int(0.25*w)
		br1 = btm_right[1] + int(0.25*h)
		buf = img[tl1:br1,tl0:br0]
		# print(tl1,br1,tl0,br0)
		if tl1<0 or tl0<0 or br1>img.shape[0] or br0>img.shape[1]:
			continue
		try:
			buf = cv2.resize(buf,(32,32))
		except:
			continue
		x = (tl0+br0)//2
		y = (tl1+br1)//2
		img_coord.append([buf,[x,y,w,h]])
	return img_coord

def filter_valid_coord(coords,veri_result,scrs):
	res = []
	scr_res = []
	for i in range(len(coords)):
		# if the verification result is 1, then append to result. 
		# filter both the coordination and scores for further non_max_suppresion
		if veri_result[i]==1:
			res.append(coords[i])
			scr_res.append(scrs[i])
	return res,scr_res

def get_iou(inp1,inp2):
	x1,y1,w1,h1 = inp1[0],inp1[1],inp1[2],inp1[3]
	x2,y2,w2,h2 = inp2[0],inp2[1],inp2[2],inp2[3]
	#print y1,y2,h1,h2
	xo = min(abs(x1+w1/2-x2+w2/2), abs(x1-w1/2-x2-w2/2))
	yo = min(abs(y1+h1/2-y2+h2/2), abs(y1-h1/2-y2-h2/2))
	if abs(x1-x2) > (w1+w2)/2 or abs(y1-y2) > (h1+h2)/2:
		return 0
	if abs(float((x1-x2)*2)) < abs(w1-w2):
		xo = min(w1, w2)
	if abs(float((y1-y2)*2)) < abs(h1-h2):
		yo = min(h1, h2)
	overlap = xo*yo
	total = w1*h1+w2*h2-overlap
	#print 'ovlp',overlap
	#print 'ttl',total
	return float(overlap)/total

def non_max_sup(coords,scr):
	# recursively get the max score in open list and delete the overlapped areas which is more than threshold
	non_max_thresh = 0.05
	open_coords = list(coords)
	open_scr = list(scr)
	result_coords = []
	
	while len(open_scr)>0:
		max_ind = np.argmax(np.array(open_scr))
		max_coord = open_coords[max_ind]
		result_coords.append(max_coord)
		del open_coords[max_ind]
		del open_scr[max_ind]
		#print len(open_scr)
		for i in range(len(open_scr),0,-1):
			iou = get_iou(open_coords[i-1],max_coord)
			#print iou
			if iou>non_max_thresh:
				del open_coords[i-1]
				del open_scr[i-1]
	return result_coords

def draw(img,coords):
	buff_img = img.copy()
	for i in coords:
		x,y,w,h = i
		cv2.rectangle(buff_img,(x-w//2,y-h//2),(x+w//2,y+h//2),(0,255,0),2)
	cv2.imshow('result',buff_img)
	cv2.waitKey(0)

# set output

b0,b1,b2,c0,c1,c2 = netpart.model_out

# set and load session
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
M.loadSess('./modelveri_tiny/',sess)

import time 

def get_coord_from_detection(img):
	areas = pre_process(img)
	res = crop(areas,img)
	if len(res)==0:
		return []
	cropped_imgs = [k[0] for k in res]
	coords = [k[1] for k in res]

	# get score and output
	veri_output = sess.run(net_veri.output,feed_dict={net_veri.inputholder:cropped_imgs})
	veri_classi = np.argmax(veri_output,1)

	veri_output = veri_output[:,1]

	valid_coord,veri_output = filter_valid_coord(coords,veri_classi,veri_output)
	valid_coord = non_max_sup(valid_coord,veri_output)
	draw(img,valid_coord)
	return valid_coord

#if __name__=='__main__':
#	t1 = time.time()
#	img = cv2.imread('5.png')
#	# for i in range(100):
#	res = get_coord_from_detection(img)
#	t2 = time.time()
#	# print(t2-t1)
