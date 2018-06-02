import numpy as np
import cv2
import conv

font = cv2.FONT_HERSHEY_SIMPLEX

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

# def get_nine_grids(rects):
# 	# duplicate the rects for in-function process

# main process
img_ori = cv2.imread('c.jpg')
img = cv2.resize(img_ori,(1300,1300))
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_gray = cv2.GaussianBlur(img_gray,(5,5),0)

ret3, th_img = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

_, ctrs, hier = cv2.findContours(th_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

side_rects = []

hier = hier[0]
center = 0

for idx in range(len(hier)):
    if hier[idx][2] == -1:
        ctr = ctrs[idx]
        area = cv2.contourArea(ctr)
        x,y,w,h = cv2.boundingRect(ctr)
        rect_area = w*h
        extent = float(area)/rect_area
        if extent > 0.8:
            side_rects.append([x,y,w,h])
            center += x
        pass

center = center / len(side_rects)
rects = side_rects
sr_tl = [0,2000]
sr_bl = [0,0]
sr_tr = [0,2000]
sr_br = [0,0]
rect3 = rects[0:4]
for rect in side_rects:
    if rect[0] < center:
        if rect[1] < sr_tl[1]:
            sr_tl = rect[0:2]
            rect3[0] = rect

        if rect[1] > sr_bl[1]:
            sr_bl = rect[0:2]
            rect3[1] = rect

    elif rect[0] > center:
        if rect[1] < sr_tr[1]:
            sr_tr = rect[0:2]
            rect3[2] = rect

        if rect[1] > sr_br[1]:
            sr_br = rect[0:2]
            rect3[3] = rect

pts1 = np.float32([sr_tl,sr_tr,sr_bl,sr_br])
pts2 = np.float32([[50, 280],[1390, 280],[50, 800],[1390, 800]])

M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img,M,(1580,1050))
#dst = cv2.resize(dst,(960,640))

for rect in rects:
    # Draw the rectangles
    cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    pass

# cv2.imshow('aaa',th_img)
# cv2.imshow('bbb',img)
cv2.imshow('ccc', dst)

digits_rect = [(280,275),(650,275),(1020,275),(280,495),(650,495),(1020,495),(280,715),(650,715),(1020,715)]
# recognize digits here
digit_imgs = []
for x,y in digits_rect:
    buf = dst[y:y+160,x:x+280]
    buf = cv2.cvtColor(buf,cv2.COLOR_BGR2GRAY)
    buf = cv2.resize(buf,(24,24))
    zeros = np.ones([28,28],dtype=np.uint8) * 255
    zeros[3:3+24,3:3+24] = buf
    buf = zeros
    buf = buf.reshape([-1])
    buf = 255 - buf
    buf = np.float32(buf) / 255.
    digit_imgs.append(buf)

scr = conv.get_pred(digit_imgs)
print(scr)

for coord,s in zip(digits_rect,scr):
    x,y = coord
    cv2.rectangle(dst,(x,y),(x+280,y+160),(0,255,0),3)
    cv2.putText(dst,str(s),(x,y+50), font, 2,(0,255,0),2,cv2.LINE_AA)

# cv2.imshow('aaa',th_img)
cv2.imshow('bbb',cv2.resize(img,(800,600)))
cv2.imshow('ddd', cv2.resize(dst,(800,600)))

cv2.waitKey(0)
