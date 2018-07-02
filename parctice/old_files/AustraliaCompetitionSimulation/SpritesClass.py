import pygame
import Functions
import math

class Player:
    """
    This is the player class
    """
    def __init__(self, plyId, x, y, framerate, scale, teammates):
        """
        Set up the player on creation
        """
        # Define the parameters of the players
        fbSpeed = 100.0 # Forward/Backward Speed in [cm/s]
        lrSpeed = 100.0 # Left/Right Speed in [cm/s]
        rotSpeed = 90.0 # Rotation Speed in [degree/s]
        radius = 30 # The radius of the robot in [cm]

        # Adjust the unit based on the scale and framerate
        self._fbSpeed = fbSpeed / (scale * framerate)
        self._lrSpeed = lrSpeed / (scale * framerate)
        self._rotSpeed = rotSpeed / (framerate)
        self._radius = radius / scale

        # Define the initial position of the player
        self._x = x
        self._y = y
        self._rot = 0 # 0 degree is facing upward

        self._dx = 0
        self._dy = 0

        self._plyId = plyId

        self._hp = 3

        self._teammates = teammates

    def getPosition(self):
        return (self._x, self._y)

    def getTeamMates(self):
        return self._teammates

    def getId(self):
        return self._plyId

    def getRadius(self):
        return self._radius

    def getRotation(self):
        return self._rot

    def getHit(self):
        self._hp -= 1

    def isDead(self):
        return self._hp <= 0

    def update(self, actionTuple, obstacleList, arenaWidth, arenaHeight):
        drot = self._rotSpeed * (actionTuple[6] - actionTuple[5])
        self._rot += drot
        if self._rot > 360: self._rot -= 360
        if self._rot < 0: self._rot += 360
        dx = self._lrSpeed * (actionTuple[4] - actionTuple[3])
        dy = self._fbSpeed * (actionTuple[2] - actionTuple[1])
        #dx, dy = Functions.rotateVector(dx, dy, self._rot)
        self._x += dx
        self._y += dy
        self._dx = dx
        self._dy = dy
        if self._x < self._radius or self._x > arenaWidth - self._radius:
            self._x -= self._dx
            self._dx = 0
        if self._y < self._radius or self._y > arenaHeight - self._radius:
            self._y -= self._dy
            self._dy = 0

        for obstacle in obstacleList.getList():
            left, top, right, bottom = obstacle.getEdge()
            if not (self._x + self._radius < left or  self._x - self._radius > right):
                if not (self._y + self._radius < top or  self._y - self._radius > bottom):
                    self._y -= self._dy
                    self._dy = 0
                    self._x -= self._dx
                    self._dx = 0

    def cancelUpdate(self):
        self._x -= self._dx
        self._y -= self._dy
        self._dx = 0
        self._dy = 0



class Bullet:
    """
    This is the bullet class
    """
    def __init__(self, shooter):
        self._speed = 20 # Speed of the bullet in pixels/frame
        self._radius = 5
        dx = 0
        dy = -(self._radius + shooter.getRadius())
        dx, dy = Functions.rotateVector(dx, dy, shooter.getRotation())
        self._x = int(shooter.getPosition()[0] + dx)
        self._y = int(shooter.getPosition()[1] + dy)
        self._shooter = shooter
        self._rot = shooter.getRotation()

    def getShooter(self):
        return self._shooter

    def getPosition(self):
        return (self._x, self._y)

    def getRadius(self):
        return self._radius

    def update(self, obstacleList, arenaWidth, arenaHeight):
        dx = 0
        dy = - self._speed
        dx, dy = Functions.rotateVector(dx, dy, self._rot)
        self._x += dx
        self._y += dy
        if self._x < self._radius or self._x > arenaWidth - self._radius:
            return -1
        if self._y < self._radius or self._y > arenaHeight - self._radius:
            return -1
        for obstacle in obstacleList.getList():
            left, top, right, bottom = obstacle.getEdge()
            if not (self._x + self._radius < left or  self._x - self._radius > right):
                if not (self._y + self._radius < top or  self._y - self._radius > bottom):
                    return -1

class RectangleObstacle:
    def __init__(self, topleft, bottomright):
        self._topleft = topleft
        self._bottomright = bottomright

    def getVertex(self):
        return self._topleft, self._bottomright

    def getEdge(self):
        """
        Output the coordinate of the left, top, right, bottom edge
        """
        return self._topleft[0], self._topleft[1], self._bottomright[0], self._bottomright[1]

class PlayerList:
    def __init__(self):
        self._list = [None, ]
        self._killed = [None, ]
        self._amount = 0
    def append(self, newPlayer):
        self._list.append(newPlayer)
        self._killed.append(0)
        self._amount += 1
    def remove(self, player):
        self._list.remove(player)
        self._amount -= 1
    def killed(self, player):
        self._list.remove(player)
        self._amount -= 1
        self._killed[player.getId()] = 1
    def getKilled(self):
        if self._amount == 0:
            return 0
        else:
            return self._killed[1:]
    def getList(self):
        if self._amount == 0:
            return 0
        else:
            return self._list[1:]

class BulletList:
    def __init__(self):
        self._list = [None, ]
        self._amount = 0
    def append(self, newBullet):
        self._list.append(newBullet)
        self._amount += 1
    def updateAll(self, obstacleList, arenaWidth, arenaHeight):
        if self._amount != 0:
            for bullet in self._list[1:]:
                if bullet.update(obstacleList, arenaWidth, arenaHeight) == -1:
                    self._list.remove(bullet)
                    self._amount -= 1
    def remove(self, bullet):
        self._list.remove(bullet)
    def getList(self):
        if self._amount == 0:
            return 0
        else:
            return self._list[1:]

class ObstacleList:
    def __init__(self):
        self._list = [None, ]
        self._amount = 0

    def append(self, newObstacle):
        self._list.append(newObstacle)
        self._amount += 1

    def getList(self):
        if self._amount == 0:
            return 0
        else:
            return self._list[1:]
