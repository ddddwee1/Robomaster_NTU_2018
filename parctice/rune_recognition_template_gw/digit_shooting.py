import rune_recog_template
import conv
import cv2
import time
import numpy as np
# import data_retriver
# import robot_prop
# import time
# import util
# import detection_mod
# from camera_module import camera_thread
# import cv2
# import sys, select, termios, tty
# import math

# def getKey():
# 	tty.setraw(sys.stdin.fileno())
# 	rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
# 	if rlist:
# 		key = sys.stdin.read(1)
# 	else:
# 		key = ''

# 	termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
# 	return key

# settings = termios.tcgetattr(sys.stdin)
# termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FPS,60)
#fps = cap.get(cv2.CAP_PROP_FPS)
#print 'fps',fps
WHITE = (255,255,255)
BLACK = (0,0,0)

image_width = 1024
image_height = 768

cap.set(3, image_width);
cap.set(4, image_height);

cap.set(14, 0.0)  #exposure
cap.set(10, 0.05) #brightness

#exp3 = cap.get(10)
#print exp3

while True:


    leftRect = []
    rightRect = []
    left_rect = []
    right_rect = []
    row_left_rect = []
    row_right_rect = []
    #img = cv2.imread('b.jpg',3)
    _,image = cap.read()



    # graycale, blurring it, and computing an edge map
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 3, 133, 133)
    edged = cv2.Canny(blurred, 120, 240,L2gradient=True)


    cv2.imshow('Original Image',image)

    cv2.waitKey(1)
    _, contours, hierarchy = cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        peri = cv2.arcLength(contour,True)
        contour = cv2.approxPolyDP(contour, 0.03 * peri, True)
        if cv2.contourArea(contour)<500 or cv2.contourArea(contour)>1500:
            continue

        x,y,w,h = cv2.boundingRect(contour)

        if (w / h) < 1.0 or (w / h) > 2.5:
            continue
        #print cv2.contourArea(contour)
        cv2.drawContours(edged, [contour], -1, (255, 255, 255), 3)

        if x < (image_width//2):
            leftRect.append(contour)
        else:
            rightRect.append(contour)

    if len(leftRect)< 5 or len(rightRect)< 5 :
        continue

    find_left = True

    for i in range(len(leftRect)):
        xi,_,_,_ = cv2.boundingRect(leftRect[i])
        for j in range(len(leftRect)):
            j1=4-j
            xj1,_,_,_ = cv2.boundingRect(leftRect[j1])
            if abs(xi - xj1) > 100:
                find_left = False

        if find_left == True:
            left_rect.append(leftRect[i])

        find_left = True

    for i in range(len(left_rect)):
        x1,y1,w1,h1 = cv2.boundingRect(left_rect[i])
        row_left_rect.append(y1)
        #cv2.rectangle(image,(x1,y1),(x1+w1,y1+h1),(0,255,0),2)

    find_right = True

    for i in range(len(rightRect)):
        xi,_,_,_ = cv2.boundingRect(rightRect[i])
        for j in range(len(rightRect)):
            j1=4-j
            xj1,_,_,_ = cv2.boundingRect(rightRect[j1])
            if abs(xi - xj1) > 100:
                find_right = False

        if find_right == True:
            right_rect.append(rightRect[i])

        find_right = True

    for i in range(len(right_rect)):
        x2,y2,w2,h2 = cv2.boundingRect(right_rect[i])
        row_right_rect.append(y2)
        #cv2.rectangle(image,(x2,y2),(x2+w2,y2+h2),(0,255,0),2)

    if len(row_left_rect) == 0 or len(row_right_rect) == 0:
        continue

    #print row_left_rect

    tl_index = np.argmin(row_left_rect)
    bl_index = np.argmax(row_left_rect)
    tr_index = np.argmin(row_right_rect)
    br_index = np.argmax(row_right_rect)

    x1,y1,w1,h1=cv2.boundingRect(left_rect[tl_index])
    x2,y2,w2,h2=cv2.boundingRect(left_rect[bl_index])
    x3,y3,w3,h3=cv2.boundingRect(right_rect[tr_index])
    x4,y4,w4,h4=cv2.boundingRect(right_rect[br_index])

    sr_tl = [x1,y1]
    sr_bl = [x2,y2+h2]
    sr_tr = [x3+w3,y3]
    sr_br = [x4+w4,y4+h4]


    pts1 = np.float32([sr_tl,sr_tr,sr_bl,sr_br])
    pts2 = np.float32([[0, 50],[300, 50],[0, 200],[300, 200]])
    #	pts2 = pts2 + np.float32([70, 35])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    inverse_M = cv2.getPerspectiveTransform(pts2,pts1)

    dst = cv2.warpPerspective(image,M,(300,200))


    digits_rect = [(51,54),(125,54),(200,54),(51,109),(125,109),(200,109),(51,163),(125,163),(200,163)]
    digit_imgs = []
    abc = 0
    for x,y in digits_rect:
        #cv2.rectangle(dst,(x,y),(x+50,y+34),(0,0,255),1)
        buf =  dst[y:y+32,x:x+50]
        buf = cv2.cvtColor(buf,cv2.COLOR_BGR2GRAY)
        #buf = cv2.equalizeHist(buf)
        #buf = cv2.fastNlMeansDenoisingColored(buf,None,10,10,7,21)
        _,buf = cv2.threshold(buf,20,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #buf = cv2.adaptiveThreshold(buf,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,33,1)
        #buf = cv2.copyMakeBorder(buf,1, 1, 1, 1, cv2.BORDER_CONSTANT, value=BLACK)
        buf = cv2.resize(buf,(24,24))
        buf = cv2.copyMakeBorder(buf, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=WHITE)
        buf = buf.reshape([-1])
        buf = 255 - buf
        buf = np.float32(buf) / 255.
        #cv2.imshow('bb',buf)
        #cv2.waitKey(0)
        buf = buf.reshape([-1])
        digit_imgs.append(buf)

    handwritten_num_raw = conv.get_pred(digit_imgs)
    print(handwritten_num_raw)

    handwritten_dict = {}
    num_7seg_dict = {}

	# filter handwritten_num
    for i in range(len(handwritten_num_raw)):
        handwritten_dict[handwritten_num_raw[i]] = np.count_nonzero(handwritten_num_raw[i])

    if len(handwritten_dict) >= 9 :
        handwritten_num= handwritten_num_raw
        print handwritten_num
    else:
        continue

    warped_digit_coords = []
    #center pixel of the digit boxes
    for digit_top_left_coord in digits_rect:
        coord = (digit_top_left_coord[0]+25,digit_top_left_coord[1]+16)
        warped_digit_coords.append(coord)

    original_digit_coords = cv2.perspectiveTransform(warped_digit_coords,inverse_M)

    #Mark the 4 corners in the original image
    cv2.circle(image,sr_tl,(0,255,0),-1)
    cv2.circle(image,sr_tr,(0,255,0),-1)
    cv2.circle(image,sr_bl,(0,255,0),-1)
    cv2.circle(image,sr_br,(0,255,0),-1)

    #Mark the 4 corners in the warped image
    cv2.circle(dst,sr_tl,(0,255,0),-1)
    cv2.circle(dst,sr_tr,(0,255,0),-1)
    cv2.circle(dst,sr_bl,(0,255,0),-1)
    cv2.circle(dst,sr_br,(0,255,0),-1)

    #Mark the 9 digits in the original image
    for original_digit_coord in original_digit_coords:
        cv2.circle(image,original_digit_coord,5,(0,0,255),-1)

    #Mark the 9 digits in the warped image
    for warped_digit_coord in warped_digit_coords:
        cv2.circle(dst,warped_digit_coord,5,(0,0,255),-1)

    cv2.imshow('Warped Image', dst)
    cv2.imshow('Original Image',image)
	# cv2.imshow('b',edged)


    cv2.waitKey(0)


