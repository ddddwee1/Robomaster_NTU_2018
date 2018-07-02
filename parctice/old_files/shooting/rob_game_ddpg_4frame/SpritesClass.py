import pygame
import Functions
import math

class Player():
    """
    This is the player class
    """
    def __init__(self, x, y, rot, plyId):
        """
        Set up the player on creation
        """
        # Call the parent class (Sprite) constructor
        super().__init__()

        # Define the speed parameters of the players
        self._fbSpeed = 2 # Forward/Backward Speed in pixels/frame
        self._lrSpeed = 1 # Left/Right Speed in pixels/frame
        self._rotSpeed = 2 # In degrees/frame

        # Define the initial position of the player
        self._x = x
        self._y = y
        self._rot = rot # 0 degree is facing upward
        self._radius = 10

        self._dx = 0
        self._dy = 0

        self._plyId = plyId

        self._hp = 3

    def getPosition(self):
        return (self._x, self._y)

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

    def update(self, actionTuple):
        drot = self._rotSpeed * (actionTuple[6] - actionTuple[5])
        self._rot += drot
        if self._rot > 360: self._rot -= 360
        if self._rot < 0: self._rot += 360
        dx = self._lrSpeed * (actionTuple[4] - actionTuple[3])
        dy = self._fbSpeed * (actionTuple[2] - actionTuple[1])
        dx, dy = Functions.rotateVector(dx, dy, self._rot)
        self._x += dx
        self._y += dy
        self._dx = dx
        self._dy = dy
        if self._x < self._radius or self._x > 500 - self._radius:
            self._x -= dx
            self._y -= dy
            self._dx = 0
            self._dy = 0
        if self._y < self._radius or self._y > 500 - self._radius:
            self._x -= dx
            self._y -= dy
            self._dx = 0
            self._dy = 0

    def cancelUpdate(self):
        self._x -= self._dx
        self._y -= self._dy
        self._dx = 0
        self._dy = 0



class Bullet():
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

    def update(self):
        dx = 0
        dy = - self._speed
        dx, dy = Functions.rotateVector(dx, dy, self._rot)
        self._x += dx
        self._y += dy
        if self._x < self._radius or self._x > 500 - self._radius:
            return -1
        if self._y < self._radius or self._y > 500 - self._radius:
            return -1

class PlayerList:
    def __init__(self):
        self._list = [None, ]
        self._amount = 0
    def append(self, newPlayer):
        self._list.append(newPlayer)
        self._amount += 1
    def remove(self, player):
        self._list.remove(player)
        self._amount -= 1
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
    def updateAll(self):
        if self._amount != 0:
            for bullet in self._list[1:]:
                if bullet.update() == -1:
                    self._list.remove(bullet)
                    self._amount -= 1
    def remove(self, bullet):
        self._list.remove(bullet)
    def getList(self):
        if self._amount == 0:
            return 0
        else:
            return self._list[1:]
