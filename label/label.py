import numpy
import cv2
import imghdr
import os

drawing = False
drawing_starting_point = (0,0)
current_image = []
working_image = []
points = []

def read_absolute_image_path(path):
    path_content = [ret_path for ret_path in os.listdir(path)]
    absolute_path = [os.path.join(path,rel_path) for rel_path in path_content]
    image_files = [file for file in absolute_path if os.path.isfile(file)] #and imghdr.what(file)]
    return image_files

def draw_rectangle(event, x, y, flags, param):
    global drawing
    global drawing_starting_point
    global current_image
    global working_image
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        drawing_starting_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        print("BUTTON UP")
        cv2.rectangle(current_image, drawing_starting_point, (x, y), (0, 255, 0), 2)
        cv2.imshow('select', current_image)
        points += points_to_top_left_width_height((drawing_starting_point,(x,y)))
    else:
        if drawing:
            working_image = current_image.copy()
            cv2.rectangle(working_image, drawing_starting_point, (x, y), (0, 255, 0), 2)
            cv2.imshow('select', working_image)

def points_to_top_left_width_height(points):
    point_1, point_2 = points
    x_1, y_1 = point_1
    x_2, y_2 = point_2
    x = max(x_1, x_2)
    y = max(y_1, y_2)
    w = abs(x_1 - x_2)
    h = abs(y_1 - y_2)
    return [str(i) for i in [x,y,w,h]]


working_path = "C:/Users/user/Desktop/Image1"
if not os.path.exists(os.path.join(working_path,'image_list.txt')):
    images_list = read_absolute_image_path(working_path)
    with open(os.path.join(working_path,'image_list.txt'),'w') as images_file:
        images_file.write("\n".join(images_list)+"\n")
else:
    with open(os.path.join(working_path,'image_list.txt'),'r') as images_file:
        images_list = [image[:-1] for image in images_file.readlines()]
cv2.namedWindow('select')
cv2.setMouseCallback('select',draw_rectangle)



while len(images_list)>0:
    points= []
    current_image_path = images_list.pop(0)
    current_image = cv2.imread(current_image_path)
    cv2.imshow('select',current_image)
    command = cv2.waitKey(0)
    print(command)
    if command == 13:
        with open(os.path.join(working_path,'annotation.txt'),'a') as annotation_file:
            annotation_file.write(" ".join([current_image_path,' '.join(points),'\n']))
        continue
    elif command == 114:
        images_list.insert(0,current_image_path)
        continue
    else:
        images_list.insert(0, current_image_path)
        with open(os.path.join(working_path,'image_list.txt'),'w') as images_file:
            images_file.write("\n".join(images_list) + "\n")
        break
