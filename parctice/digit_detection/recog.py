import numpy as np 
import cv2
import conv

SMALL_SUPPRESSION = 1000
LARGE_SUPPRESSION = 10000

def get_iou(inp1,inp2):
	x1,y1,w1,h1 = inp1[0],inp1[1],inp1[2],inp1[3]
	x2,y2,w2,h2 = inp2[0],inp2[1],inp2[2],inp2[3]
	x1 = x1 + w1/2
	y1 = y1 + h1/2
	x2 = x2 + w2/2
	y2 = y2 + h2/2
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

def filter_abnormal_ratio(rects):
	res = []
	for x,y,w,h in rects:
		if float(w)/float(h)<1/6 or float(w)/float(h)>2.:
			continue
		else:
			res.append([x,y,w,h])
	return res 

def remove_overlap(rects):
	openset = list(rects) # openset for raw data
	result = [] # for filtered data
	while len(openset)!=0:
		current = openset[-1]
		for i in range(len(openset),0,-1):
			iou = get_iou(current,openset[i-1])
			if iou>0.05:
				del openset[i-1]
		result.append(current)
	print(len(result))
	return result

# def get_nine_grids(rects):
# 	# duplicate the rects for in-function process
	

# main process 
img = cv2.imread('b.jpg')
img = cv2.resize(img,(1300,1300))
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(img,5)

th_img = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,21,9)

_, ctrs, hier = cv2.findContours(th_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

rects = [cv2.boundingRect(ctr) for ctr in ctrs]

rects2 = []
for rect in rects:
	if rect[2]*rect[3] < SMALL_SUPPRESSION or rect[2]*rect[3] > LARGE_SUPPRESSION:
		continue
	else:
		rects2.append(rect)

rects = rects2

rects = filter_abnormal_ratio(rects)
rects = remove_overlap(rects)

for rect in rects:
    # Draw the rectangles
    cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 

cv2.imshow('aaa',th_img)
cv2.imshow('bbb',img)
cv2.waitKey(0)
