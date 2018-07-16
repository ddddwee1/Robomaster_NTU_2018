import numpy as np
import cv2
import conv

digit_imgs = []
for i in range(1,10):
    img = cv2.imread('./DigitImages/Flaming_Digits/%d.jpg'%(i))
    img2 = np.zeros(img.shape[:2], dtype=np.uint8)
    img = img[:,40:240,0]
    ret, img = cv2.threshold(img,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
    _, cnts, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img2, cnts, -1, 255, -1)
    img2 = img2[:,:200]
    img = img2

    buf = cv2.resize(img, (28,28))
    buf = cv2.bitwise_not(buf)
    buf = buf.reshape([-1])
    buf = 255 - buf
    buf = np.float32(buf) / 255.
    digit_imgs.append(buf)
    scr = conv.get_pred(digit_imgs)
    print(scr)

    cv2.imshow('', img)
    cv2.waitKey(0)
