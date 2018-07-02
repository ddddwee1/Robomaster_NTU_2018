import numpy as np
import cv2
import conv

digit_imgs = []
for i in range(1,10):
    img = cv2.imread('./DigitImages/Flaming_Digits/%d.jpg'%(i))
    img2 = np.zeros(img.shape[:2], dtype=np.uint8)
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = img[:,:,0]
    #img = cv2.GaussianBlur(img, (9,9), 0)
    ret, img = cv2.threshold(img,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
    #kernel = np.ones((3,3),np.uint8)
    #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    #img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    #img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    #img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

    #ret, img = cv2.threshold(img,0,255,cv2.THRESH_TRUNC+cv2.THRESH_OTSU)

    #img = cv2.bitwise_not(img)
    _, cnts, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, cnts2, hier2 = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnts3 = []
    #for cnt in cnts:
    #    cnts3.append(cv2.approxPolyDP(cnt, 1, True))
    cv2.drawContours(img2, cnts, -1, 255, -1)
    """
    for i in range(len(cnts2)):
        c = cnts2[i]
        h = hier2[0][i]
        if h[2] == -1:
            #cv2.drawContours(img2, [c], 0, 255, -1)
            circle = cv2.minEnclosingCircle(c)
            x = int(circle[0][0])
            y = int(circle[0][1])
            r = int(circle[1])
            #cv2.circle(img2, (x,y), r, 0)
    """
    img = img2
    """
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,10,param1=50,param2=30,minRadius=0,maxRadius=0)
    if circles != None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
            #img = cv2.erode(img, kernel, iterations = 1)
    """
    """
    skel = np.zeros(img.shape, dtype=np.uint8)
    done = False
    while not done:
        temp = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        temp = cv2.bitwise_not(temp)
        temp = cv2.bitwise_and(img, temp)
        skel = cv2.bitwise_or(temp, skel)
        img = cv2.erode(img, kernel, iterations = 1)
        if np.amax(img) == 0:
            done = True
    """

    #img = cv2.erode(img,kernel,iterations = 2)
    #skel = cv2.bitwise_not(skel)
    buf = cv2.resize(img, (24,24))
    buf = cv2.bitwise_not(buf)
    buf = cv2.copyMakeBorder(buf, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255,255,255))
    buf = buf.reshape([-1])
    buf = 255 - buf
    buf = np.float32(buf) / 255.
    digit_imgs.append(buf)
    scr = conv.get_pred(digit_imgs)
    print(scr)

    cv2.imshow('', img)
    cv2.waitKey(0)
