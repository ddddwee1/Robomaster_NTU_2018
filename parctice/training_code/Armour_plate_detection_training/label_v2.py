import numpy
import cv2
import imghdr
import os
import platform #for gimmicks

drawing = False
drawing_starting_point = (0,0)
current_image = []
working_image = []
points = []


def read_absolute_image_path(path):
    """
    Take in the path of the folder containing the images and return a list of the absolute paths of the images

    Parameters
    ----------
    path: string
        path of the folder containing the images

    Returns
    -------
    image_files: list
        list of the absolute paths of the images
    """
    path_content = [ret_path for ret_path in os.listdir(path)]   #list of file names in the path
    absolute_path = [os.path.join(path,rel_path) for rel_path in path_content]   #absolute paths of all of the objects in the path
    image_files = [file for file in absolute_path if os.path.isfile(file) and file.endswith(image_format)]# and imghdr.what(file)]
    return image_files

def draw_rectangle(event, x, y, flags, param):
    """
    Event callback(draw rectangle)
    """
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
        # print("BUTTON UP")
        cv2.rectangle(current_image, drawing_starting_point, (x, y), (0, 255, 0), 1)
        cv2.imshow('select', current_image)
        if abs(x-drawing_starting_point[0]) <= 10 or abs(y-drawing_starting_point[1]) <= 10:
            if platform == 'Linux':
                os.system('spd-say "Very Small bounding box detected"')
            print"Warning! Very small bounding box detected! You might want to check your bounding box"
        points += points_to_top_left_width_height(  (drawing_starting_point,(x,y)))
    else:
        if drawing:
            working_image = current_image.copy()
            cv2.rectangle(working_image, drawing_starting_point, (x, y), (0, 0, 255), 1)
            cv2.imshow('select', working_image)

def points_to_top_left_width_height(points):
    point_1, point_2 = points
    x_1, y_1 = point_1
    x_2, y_2 = point_2
    x = max(x_1, x_2) #btm right x coordinate of the bounding box
    y = max(y_1, y_2) #btm right y coordintate of the bounding box
    w = abs(x_1 - x_2) #width of the bounding box
    h = abs(y_1 - y_2) #height of the bounding box
    return [str(i) for i in [x,y,w,h]]


def write_to_file(annotation_path,image_number, xywh_list,image_format):
    """
    Given the image number, update the xywh list for that image

    Parameters
    ----------
    annotation_path: string
        path of annotation.txt
    image number: int
        index of the image
    xywh_list: list
        list of strings (eg. ['1','2','3','4'])
    image_format: string
        image format(eg. '.png', '.jpg')

    Returns
    -------
    sucess: bool
        True if success
    """
    with open(annotation_path, 'r') as file:
        data = file.readlines()
    line = data[image_number -1].strip()
    image_path = line.split(image_format)[0] + image_format
    new_data = " ".join([image_path]+xywh_list)+ "\n"
    data[image_number -1] = new_data

    with open(annotation_path, 'w') as writefile:
        writefile.writelines(data)

def read_from_file(annotation_path,image_number,image_format):
    """
    Given the image number, return the list of coords for that image

    Parameters
    ----------
    annotation_path: string
        path of the annotation.txt
    image_number: int
        index of the image
    image_format: string
        image format(eg. '.png', '.jpg')
    Returns
    -------
    image_path: string
        image path
    xywh_list: list
        list of [x,y,w,h]. Each [x,y,w,h] corresponds to a bounding box. Returns an empty list if there are no bounding boxes

    """
    xywh_list = []
    with open(annotation_path, 'r') as fp:
        for i, line in enumerate(fp):
            if i == image_number-1:
                data = line.strip()               #remove newlines
                data = data.split(image_format)   #separate image path the bounding boxes's xywh
                image_path = data[0] + image_format
                coord_data = data[1]
                coord_data = coord_data[1:] #remove the space
                if len(coord_data) >1:
                    coord_data = coord_data.split(" ")
                    temp = []
                    for j,coord in enumerate(coord_data):
                        # print"coord =",coord
                        if (j+1)%4 != 0:
                            temp.append(int(coord))
                        else:
                            temp.append(int(coord))
                            xywh_list.append(temp)
                            temp = []
                break
    return image_path, xywh_list


def xywh_to_corner_coords(xywh):
    """
    Convert x,y,w,h to the 2 corner coordinates for drawing rectangle

    Parameters
    ----------
    xywh: list
        a list containing the x,y,w,h of a bounding box

    Returns
    -------
    x1, y1, x2, y2: int
        coordinates of the 2 corners(top left and btm right of the rectangle)
    """
    x,y,w,h = xywh
    x2 = x
    y2 = y
    x1 = x2 - w
    y1 = y2 - h
    return x1,y1,x2,y2

#Parameters to change
working_path = '/home/kyubey/Desktop/labeling/red videos' #Folder which contains the images
image_format = '.jpg' #Image format of the images inside the folder specified in working path
#Initialisation
cv2.namedWindow('select')
cv2.setMouseCallback('select',draw_rectangle)

image_number = 1 #start from number 1
annotation_list = []
#for sound alerts :)
#Check platform
platform = platform.system()
if platform == 'Linux':
    print "Hi pls install this for special sound effects using this command:\nsudo apt install speech-dispatcher"
print"\n~INSTRUCTIONS~"
print"Change the image format and the working path variables in the source code before running the script"
print"Press Enter to save bounding boxes of current image and proceed to next image"
print"Press A to go back to previous image"
print"Press Esc to exit without saving current image"
print"Press R to clear bounding boxes for current image"


#if annotation.txt does not exist
if not os.path.exists(os.path.join(working_path,'annotation.txt')):
    annotation_list = read_absolute_image_path(working_path)
    with open(os.path.join(working_path, 'annotation.txt'),'w') as annotation_file:
        annotation_file.write("\n".join(sorted(annotation_list))+"\n")
#if annotation.txt exists
else:
    with open(os.path.join(working_path, 'annotation.txt'),'r') as annotation_file:
        annotation_list = annotation_file.readlines()


while image_number <= len(annotation_list):
    print"Image number = ",image_number, "/",len(annotation_list)
    points = []
    current_image_path, xywh_list = read_from_file(os.path.join(working_path, 'annotation.txt'), image_number,image_format)

    current_image = cv2.imread(current_image_path)
    if len(xywh_list) != 0:
        for xywh in xywh_list:
            x,y,w,h = xywh
            if w<=10 or h <=10:
                print"Warning! Very small bounding box detected. You might want to check this bounding box : ",xywh
                if platform == 'Linux':
                    os.system('spd-say "Very small bounding box detected!"')
            points += [str(i) for i in xywh]
            x1,y1,x2,y2 = xywh_to_corner_coords(xywh)
            cv2.rectangle(current_image,(x1,y1),(x2,y2),(0,255,0),1)
    cv2.imshow('select',current_image)
    command = cv2.waitKey(0)
    #Only allow pressing of valids buttons
    while command !=13 and command != 27 and command != 114 and command != 97:
        print"Please press a valid button!"
        if platform == 'Linux':
                os.system('spd-say "Invalid button"')
        command = cv2.waitKey(0)

    if command == 13:#enter to go to save bounding boxes and go to next picture
        write_to_file(os.path.join(working_path, 'annotation.txt'),image_number,points,image_format)
        if image_number == len(annotation_list):
            print"Congratulations! You have finished labelling!"
            if platform == 'Linux':
                os.system('spd-say "Congratulations! You have finished labelling!"')
        image_number += 1

    elif command == 27: #Esc to exit
        print"Exiting"
        break

    elif command == 97:#a button to go back to previous picture
        write_to_file(os.path.join(working_path, 'annotation.txt'),image_number,points,image_format)
        if image_number == 1:
            print("Cannot go to previous picture because you are currently at the first picture")
        else:
            image_number -= 1
    elif command == 114:#r button to restart
        points = []
        write_to_file(os.path.join(working_path, 'annotation.txt'),image_number,points,image_format)




