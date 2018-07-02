import numpy as np 
import datareader2
import cv2
import netpart
import net_veri
import tensorflow as tf 
import model as M 

threshold = 0.7
MAXITER = 200000

def get_img_coord(img,c,b,multip):
	# get the coordinations by c and b
	# multip is the gridsize.
	res = []
	c = c[0]
	b = b[0]
	row,col,_ = b.shape
	c = c.reshape([-1])
	ind = c.argsort()[-5:][::-1]
	for aaa in ind:
		i = aaa//col
		j = aaa%col 
		x = int(b[i][j][0])+j*multip+multip//2
		y = int(b[i][j][1])+i*multip+multip//2
		w = int(b[i][j][2])
		h = int(b[i][j][3])
		M = np.float32([[1,0,-(x-w//2)],[0,1,-(y-h//2)]])
		cropped = cv2.warpAffine(img,M,(w,h))
		cropped = cv2.resize(cropped,(32,32))
		# append [cropped_image,[x,y,w,h]] to result list
		res.append([cropped,[x,y,w,h]])
	return res 

def crop(img,bs,cs):
	# triple scales
	multi = [8,32,128]
	res = []
	for i in range(3):
		# the elements in res are [cropped_imgs, coordinates]
		buff = get_img_coord(img,cs[i],bs[i],multi[i])
		res += buff
	return res

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
	xo = min(abs(x1+w1/2-x2+w2/2), abs(x1-w1/2-x2-w2/2))
	yo = min(abs(y1+h1/2-y2+h2/2), abs(y1-h1/2-y2-h2/2))
	if abs(x1-x2) > (w1+w2)/2 or abs(y1-y2) > (h1+h2)/2:
		return 0
	if abs(x1-x2) < abs(w1-w2):
		xo = min(w1, w2)
	if abs(y1-y2) < abs(h1-h2):
		yo = min(h1, h2)
	overlap = xo*yo
	total = w1*h1+w2*h2-overlap
	return overlap/total

def non_max_sup(coords,scr):
	# recursively get the max score in open list and delete the overlapped areas which is more than threshold
	non_max_thresh = 0.3
	open_coords = list(coords)
	open_scr = list(scr)
	result_coords = []
	
	while len(open_scr)>0:
		max_ind = np.argmax(np.array(open_scr))
		max_coord = open_coords[max_ind]
		result_coords.append(open_coords[max_ind])
		for i in range(len(open_scr),0,-1):
			iou = get_iou(open_coords[i-1],max_coord)
			if iou>non_max_thresh:
				open_coords.pop(i-1)
				open_scr.pop(i-1)
	return result_coords

def draw(img,coords):
	buff_img = img.copy()
	for i in coords:
		x,y,w,h = i
		cv2.rectangle(buff_img,(x-w//2,y-h//2),(x+w//2,y+h//2),(0,255,0),2)
	cv2.imshow('result',buff_img)
	cv2.waitKey(0)

reader = datareader2.reader()
b0,b1,b2,c0,c1,c2 = netpart.model_out

with tf.Session() as sess:
	M.loadSess('./modelveri/',sess)
	for i in range(MAXITER):
		img,_ = reader.get_img()
		buff_out = sess.run([b0,b1,b2,c0,c1,c2],feed_dict={netpart.inpholder:[img]})
		bs,cs = buff_out[:3],buff_out[3:]
		res = crop(img,bs,cs)
		cropped_imgs = [k[0] for k in res]
		coords = [k[1] for k in res]
		# get the score by softmax
		veri_output = sess.run(tf.nn.softmax(net_veri.output,1),feed_dict={net_veri.inputholder:cropped_imgs})
		# get the classification result by argmax. If 0, then it is regarded as false, if 1 then true
		veri_classi = np.argmax(veri_output,1)
		# get the score of only 1 side. discard the 0 side
		veri_output = veri_output[:,1]
		# filter the results by verification results
		valid_coord,veri_output = filter_valid_coord(coords,veri_classi,veri_output)
		# apply non_max_suppresion
		valid_coord = non_max_sup(valid_coord,veri_output)
		# draw the result
		draw(img,valid_coord)