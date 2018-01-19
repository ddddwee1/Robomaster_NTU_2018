import pygame
import SpritesClass
import math
import numpy as np

def rotateVector(dx, dy, degree):
    a = math.cos(math.radians(degree))
    b = math.sin(math.radians(degree))
    c = - b
    rotM = np.array(((a, b), (c, a)))
    d = np.array((dx, dy))
    d = np.matmul(d, rotM)
    return d[0], d[1]

def intToBinList(theInt, length):
    theList = [theInt % 2, ]
    theInt >>= 1
    for i in range(1, length):
        theList.append(theInt % 2)
        theInt >>= 1
    return list(reversed(theList))

def binListToInt(theList):
    length = len(theList)
    theInt = 0
    for i in range(length):
        theInt += theList[i] * (2**(length-i-1))
    return theInt

def getEnemyDistance(player, enemy):
    ppos = player.getPosition()
    epos = enemy.getPosition()
    d = (epos[0] - ppos[0])**2 + (epos[1] - ppos[1])**2
    d = math.sqrt(d)
    return d

def getEnemyAngle(player, enemy):
    ppos = player.getPosition()
    epos = enemy.getPosition()
    prot = player.getRotation()
    prot_V = np.array((math.sin(math.radians(prot)), -math.cos(math.radians(prot))))
    protl_V = np.array((math.sin(math.radians(prot-35)), -math.cos(math.radians(prot-35))))
    epos_V = np.array((epos[0] - ppos[0], epos[1] - ppos[1]))
    ang = getAngleFromVector(prot_V, epos_V)
    if ang < 35:
        if getAngleFromVector(protl_V, epos_V) > 35:
            return ang
        else:
            return -ang
    else:
        return None

def getAngleFromVector(vec1, vec2):
    result = np.dot(vec1, vec2)
    vec1M = np.linalg.norm(vec1)
    vec2M = np.linalg.norm(vec2)
    ang = result / (vec1M * vec2M)
    ang = math.degrees(math.acos(ang))
    return ang

def logVisibleEnemyPosition(player, enemy):
    log = ""
    ang = getEnemyAngle(player, enemy)
    if ang == None:
        log = log + "0,0,"
    else:
        dis = getEnemyDistance(player, enemy)
        log = log + str(ang) + "," + str(dis) + ","
    return log

def logActionTuple(act):
    log = ""
    log = log + str(act[4] - act[3]) + ","
    log = log + str(act[1] - act[2]) + ","
    log = log + str(act[6] - act[5]) + ","
    log = log + str(act[7]) + ","
    return log
