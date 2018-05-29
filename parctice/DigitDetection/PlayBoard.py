import pygame
import random
import os
import cv2
import numpy as np

# Initialize Constant
HANDWRITTEN_IMAGE_AMOUNT = [5444, 6179, 5470, 5638, 5307, 4987, 5417, 5715, 5389, 5454]
DIGIT_LIST = [0,1,2,3,4,5,6,7,8,9]
SEVENSEGMENT_COORDINATES = ((290,20), (370,20), (450,20), (530,20), (610,20))
HANDWRITTEN_COORDINATES = ((152,160), (384,160), (616,160), (152,320), (384,320), (616,320), (152,480), (384,480), (616,480))
SEVENSEGMENT_SIZE = (60,100)
HANDWRITTEN_SIZE = (192,120)

# Initialize color variables
COLOR_WHITE = (255,255,255)
COLOR_BLACK = (0,0,0)
COLOR_GRAY = (120,120,120)

# Display Setting
DISPLAY_HEIGHT = 640
DISPLAY_WIDTH = 960

# Set up PyGame
pygame.init()
screen = pygame.display.set_mode([DISPLAY_WIDTH, DISPLAY_HEIGHT])
pygame.display.set_caption('Digits Board')

# Functions
def updateDigits(image, digitType):
    if digitType == 0:
        list = random.choices(DIGIT_LIST, k=5)

    elif digitType == 1:
        list = random.sample(DIGIT_LIST, k=10)

def updateSevenSegment(display):
    list = random.choices(DIGIT_LIST, k=5)
    # Put the digit image to the display image
    for i in range(len(list)):
        img = cv2.imread('./DigitImages/Seven_Segments/%d.jpg'%(list[i]), cv2.IMREAD_COLOR)
        coor = SEVENSEGMENT_COORDINATES[i]
        size = SEVENSEGMENT_SIZE
        display[coor[1]:coor[1]+size[1],coor[0]:coor[0]+size[0]] = img
    # Return the list at the end
    return list

def updateHandwritten(display, sevensegmentList):
    list = random.sample(DIGIT_LIST, k=10)
    # Initialize a variable to store the digit and file index of the images
    indexList = [None, ]
    # Since there are only 9 boxes for 10 digits, one of the digit must be removed from the list
    removedNumber = random.randint(0,9)
    while removedNumber in sevensegmentList:
        removedNumber = random.randint(0,9)
    list.remove(removedNumber)
    # For every digit in the list
    for i in range(len(list)):
        # Pick by random the image that represent the digit
        index = random.randrange(HANDWRITTEN_IMAGE_AMOUNT[list[i]])
        filename = str(list[i]) + '_' + str(index)
        indexList.append(filename)
        # Put the digit image to the display image
        img = cv2.imread('./DigitImages/Handwritten_%d/%d_%d.jpg'%(list[i],list[i],index), cv2.IMREAD_COLOR)
        coor = HANDWRITTEN_COORDINATES[i]
        size = HANDWRITTEN_SIZE
        display[coor[1]:coor[1]+size[1],coor[0]:coor[0]+size[0]] = img
    # Return the list at the end
    return indexList[1:]

DisplayImage = cv2.imread('DigitBoardBackground.jpg', cv2.IMREAD_COLOR)
done = False
counter = 0
clock = pygame.time.Clock()
framerate = 30

while not done:
    # Update seven segment digits
    if counter % (5*framerate) == 0:
        sevensegmentList = updateSevenSegment(DisplayImage)

    # Update handwritten digits
    if counter % framerate == 0:
        handwrittenList = updateHandwritten(DisplayImage, sevensegmentList)

    # Update pygame display
    if counter % framerate == 0:
        surface = pygame.display.get_surface()
        surfaceImage = cv2.cvtColor(DisplayImage, cv2.COLOR_BGR2RGB)
        surfaceImage = np.transpose(surfaceImage, (1, 0, 2))
        #print(surfaceImage.shape)
        #print(surface.get_size())
        pygame.surfarray.blit_array(surface, surfaceImage)
        pygame.display.flip()

    # Check for exit key
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            status = -1
        elif event.type == pygame.KEYDOWN:
            if event.key == 27: done = True

    # Update counter:
    counter = counter + 1
    counter = counter % 150

    # Set framerate
    clock.tick(framerate)

pygame.quit()
