import pygame
import Functions
import math

class Display:
    def __init__(self, owner):
        self._width = 500
        self._height = 500
        self._owner = owner
        self._screen = pygame.display.set_mode([self._width, self._height])
        self._playerColor = ((0,0,0),(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255))
        self._corners = ((self._width,0),(self._width,self._height),(0,self._height),(0,0))

    def getScreen(self):
        return self._screen

    #def drawAllPlayers(x,y,id):

    def drawPlayer(self, player):
        pos = player.getPosition()
        x = int(round(pos[0]))
        y = int(round(pos[1]))
        pygame.draw.circle(self._screen, self._playerColor[player.getId()], \
                            (x, y), player.getRadius())

    def drawBullet(self, bullet):
        pos = bullet.getPosition()
        x = int(round(pos[0]))
        y = int(round(pos[1]))
        pygame.draw.circle(self._screen, (100,100,100), \
                            (x, y), bullet.getRadius())

    def drawAllBullets(self, bulletList):
        allBullets = bulletList.getList()
        if allBullets == 0:
            pass
        else:
            for bullet in allBullets:
                self.drawBullet(bullet)

    def drawAllPlayers(self, playerList):
        allPlayers = playerList.getList()
        if allPlayers == 0:
            pass
        else:
            for player in allPlayers:
                self.drawPlayer(player)

    def maskFOV(self, player):
        pos = player.getPosition()
        rot = player.getRotation()
        dx = 0
        dy = -math.sqrt(self._width**2 + self._height**2)
        ldx, ldy = Functions.rotateVector(dx, dy, rot-35)
        rdx, rdy = Functions.rotateVector(dx, dy, rot+35)
        lx = ldx + pos[0]
        ly = ldy + pos[1]
        rx = rdx + pos[0]
        ry = rdy + pos[1]
        pointList = [pos, (rx, ry)]
        leftPointList = [pos, (pos[0], 0)]
        rightPointList = [pos, (rx, ry)]
        pygame.draw.line(self._screen, (0,0,0), pos, (lx,ly), 1)
        pygame.draw.line(self._screen, (0,0,0), pos, (rx,ry), 1)
        lrot = rot - 35
        if lrot < 0: lrot += 360
        elif lrot > 360: lrot -= 360
        rrot = rot + 35
        if rrot < 0: rrot += 360
        elif rrot > 360: rrot -= 360
        for corner in self._corners:
            dx = corner[0] - pos[0]
            dy = pos[1] - corner[1]
            ang = math.degrees(math.atan2(dx,dy))
            if ang < 0: ang += 360
            if rrot > lrot:
                if ang > 0 and ang < lrot:
                    leftPointList.append(corner)
                elif ang < 360 and ang > rrot:
                    rightPointList.append(corner)
            else:
                if ang > rrot and ang < lrot:
                    pointList.append(corner)
        if rrot > lrot:
            leftPointList.append((lx, ly))
            rightPointList.append((pos[0], 0))
            pygame.draw.polygon(self._screen, (0,0,0), leftPointList)
            pygame.draw.polygon(self._screen, (0,0,0), rightPointList)
        else:
            pointList.append((lx, ly))
            pygame.draw.polygon(self._screen, (0,0,0), pointList)

    def drawLaserSight(self, player):
        spos = player.getPosition()
        prot = player.getRotation()
        lineV = [math.sin(math.radians(prot)), -math.cos(math.radians(prot))]
        lineV[0] = math.sqrt(self._width**2 + self._height**2) * lineV[0]
        lineV[1] = math.sqrt(self._width**2 + self._height**2) * lineV[1]
        epos = (spos[0]+lineV[0], spos[1]+lineV[1])
        pygame.draw.line(self._screen, (255,0,0), spos, epos)
