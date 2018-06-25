import rune_recog_template
import conv
import cv2
import time
import numpy as np
import data_retriver
import robot_prop
import time
import util
import detection_mod
from camera_module import camera_thread
import cv2
import math


def get_handwritten_coords(image, DEBUG):
    """
    Given a raw image, returns the coordinates of the handwritten digits in the image

    Parameters
    ----------
    image: numpy array
        original image in RGB format

    Returns
    -------
    results: dict
        key: digit
        value: coord of the digit
    """
    leftRect = []
    rightRect = []
    left_rect = []
    right_rect = []
    row_left_rect = []
    row_right_rect = []


    # graycale, blurring it, and computing an edge map
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 3, 133, 133)
    edged = cv2.Canny(blurred, 120, 240,L2gradient=True)
    cv2.imshow('Original Image',image)
    cv2.waitKey(1)
    _, contours, hierarchy = cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


    #check contour area and split them into contours on left half and right half of image
    for contour in contours:
        peri = cv2.arcLength(contour,True)
        contour = cv2.approxPolyDP(contour, 0.03 * peri, True)
        if cv2.contourArea(contour)<500 or cv2.contourArea(contour)>1500:
            continue
        x,y,w,h = cv2.boundingRect(contour) #top left coord,width and height
        if (w / h) < 1.0 or (w / h) > 2.5:
            continue
        cv2.drawContours(edged, [contour], -1, (255, 255, 255), 3)
        if x < (image_width//2):
            leftRect.append(contour)
        else:
            rightRect.append(contour)


    #skip if less than 5 contours are found on 1 side of image
    if len(leftRect)< 5 or len(rightRect)< 5 :
        return


    #check alignment of the bounding boxes of contour on left half
    find_left = True
    for i in range(len(leftRect)):
        xi,_,_,_ = cv2.boundingRect(leftRect[i])
        for j in range(len(leftRect)):
            j1=4-j
            xj1,_,_,_ = cv2.boundingRect(leftRect[j1])
            if abs(xi - xj1) > 100:       #topmost rectangle and btm most rectangle x-coord max deviation
                find_left = False
        if find_left == True:
            left_rect.append(leftRect[i])
        find_left = True
    for i in range(len(left_rect)):
        x1,y1,w1,h1 = cv2.boundingRect(left_rect[i])
        row_left_rect.append(y1)


    #check alignment of the bounding boxes of contour on right half
    find_right = True
    for i in range(len(rightRect)):
        xi,_,_,_ = cv2.boundingRect(rightRect[i])
        for j in range(len(rightRect)):
            j1=4-j
            xj1,_,_,_ = cv2.boundingRect(rightRect[j1])
            if abs(xi - xj1) > 100:    #topmost rectangle and btm most rectangle x-coord max deviation
                find_right = False
        if find_right == True:
            right_rect.append(rightRect[i])
        find_right = True
    for i in range(len(right_rect)):
        x2,y2,w2,h2 = cv2.boundingRect(right_rect[i])
        row_right_rect.append(y2)


    #Skip if no boxes are detected on left half or right half
    if len(row_left_rect) == 0 or len(row_right_rect) == 0:
        return


    #find the 4 bounding boxes of the corner contours
    tl_index = np.argmin(row_left_rect)
    bl_index = np.argmax(row_left_rect)
    tr_index = np.argmin(row_right_rect)
    br_index = np.argmax(row_right_rect)
    x1,y1,w1,h1=cv2.boundingRect(left_rect[tl_index])
    x2,y2,w2,h2=cv2.boundingRect(left_rect[bl_index])
    x3,y3,w3,h3=cv2.boundingRect(right_rect[tr_index])
    x4,y4,w4,h4=cv2.boundingRect(right_rect[br_index])


    #corner points in original image
    sr_tl = [x1,y1]
    sr_bl = [x2,y2+h2]
    sr_tr = [x3+w3,y3]
    sr_br = [x4+w4,y4+h4]


    #perspective transform matrix from original to warped image
    pts1 = np.float32([sr_tl,sr_tr,sr_bl,sr_br])
    pts2 = np.float32([[0, 50],[300, 50],[0, 200],[300, 200]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(image,M,(300,200))


    #reverse perspective transform matrix from warped image to original image
    inverse_M = cv2.getPerspectiveTransform(pts2,pts1)


    #top left coords of handwritten_digits in warped image
    digits_rect = [(51,54),(125,54),(200,54),(51,109),(125,109),(200,109),(51,163),(125,163),(200,163)]


    #Recognise the digits
    digit_imgs = []
    abc = 0
    for x,y in digits_rect:
        buf =  dst[y:y+32,x:x+50]
        buf = cv2.cvtColor(buf,cv2.COLOR_BGR2GRAY)
        _,buf = cv2.threshold(buf,20,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        buf = cv2.resize(buf,(24,24))
        buf = cv2.copyMakeBorder(buf, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=WHITE)
        buf = buf.reshape([-1])
        buf = 255 - buf
        buf = np.float32(buf) / 255.
        buf = buf.reshape([-1])
        digit_imgs.append(buf)
    handwritten_num_raw = conv.get_pred(digit_imgs)
    print(handwritten_num_raw)


    #filter handwritten_dict
    handwritten_dict = {}
    for i in range(len(handwritten_num_raw)):
        handwritten_dict[handwritten_num_raw[i]] = np.count_nonzero(handwritten_num_raw[i])

    if len(handwritten_dict) >= 9 :
        handwritten_num= handwritten_num_raw
        print"Handwritten digits detected : ",handwritten_num
    else:
        return


    warped_digit_coords = []
    #center pixel of the digit boxes
    for digit_top_left_coord in digits_rect:
        coord = (digit_top_left_coord[0]+25,digit_top_left_coord[1]+16)
        warped_digit_coords.append(coord)
    warped_digit_coords = np.asarray( warped_digit_coords,dtype='float32')
    warped_digit_coords = np.array([warped_digit_coords])


    #coords of the center of the digits in the original image
    original_digit_coords = cv2.perspectiveTransform(warped_digit_coords,inverse_M) #np.array([[[x1,y1],[x2,y2]]])
    results = {}
    for i,coord in enumerate(original_digit_coords[0]):
        results[handwritten_num] = tuple(coord)


    #Draw the 4 corners in the original image
    if DEBUG == True:
        cv2.circle(image,tuple(sr_tl),5,(0,255,0),-1)
        cv2.circle(image,tuple(sr_tr),5,(0,255,0),-1)
        cv2.circle(image,tuple(sr_bl),5,(0,255,0),-1)
        cv2.circle(image,tuple(sr_br),5,(0,255,0),-1)

        #Mark the 4 corners in the warped image
        cv2.circle(dst,tuple(sr_tl),5,(0,255,0),-1)
        cv2.circle(dst,tuple(sr_tr),5,(0,255,0),-1)
        cv2.circle(dst,tuple(sr_bl),5,(0,255,0),-1)
        cv2.circle(dst,tuple(sr_br),5,(0,255,0),-1)

        #Mark the 9 digits in the original image
        for original_digit_coord in original_digit_coords[0]:
            cv2.circle(image,tuple(original_digit_coord),5,(0,0,255),-1)

        #Mark the 9 digits in the warped image
        for warped_digit_coord in warped_digit_coords[0]:
            cv2.circle(dst,tuple(warped_digit_coord),5,(0,0,255),-1)

        #Show image
        cv2.imshow('Warped Image', dst)
        cv2.imshow('Original Image',image)
        cv2.waitKey(1)

    return results


def get_7segdigits_coords(image):
    """
    Given a raw image, return the detected 7 segment digits
    """
    #TODO





