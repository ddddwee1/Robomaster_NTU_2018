import numpy as np
import cv2
import time

cap = cv2.VideoCapture(1) # video capture source camera (Here webcam of laptop) 
cap.set(3, 640);
cap.set(4, 480);

counter = 0
capture_duration = 180
start_time = time.time()

while(int(time.time() - start_time) < capture_duration ):
    ret,frame = cap.read() 
    frame_name = "c" + str(counter) + ".png"
    cv2.imwrite(frame_name,frame)
    
    cv2.imshow('img1',frame)
    cv2.waitKey(30)
    counter += 1

cv2.destroyAllWindows()
cap.release()
