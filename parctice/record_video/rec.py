import cv2 

def hist_equal(img):
	equ = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	equ[:,:,2] = cv2.equalizeHist(equ[:,:,2])
	equ = cv2.cvtColor(equ, cv2.COLOR_HSV2BGR)
	return equ

MAX_FRAME = 50000
FRAME_SKIP = 4
j = -15

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_EXPOSURE,-6.)
cap.set(10,0.01)

for i in range(MAX_FRAME):
	if i <= 0:
		continue
	_,img = cap.read()
	img = hist_equal(img)
	if i % FRAME_SKIP != 1:
		j+=1
		cv2.imwrite('./imgs/%d.jpg'%j,img) 
