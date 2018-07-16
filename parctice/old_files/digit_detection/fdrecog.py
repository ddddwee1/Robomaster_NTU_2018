import numpy as np
import cv2
import conv

digit_imgs = []
for i in range(1,10):
    img = cv2.imread('./DigitImages/Flaming_Digits/%d.jpg'%(i))
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = img[:,:,0]
    img = cv2.GaussianBlur(img, (5,5), 0)
    ret, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #kernel = np.ones((3,3),np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
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
