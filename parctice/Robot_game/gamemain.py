import pygame 
from pygame.locals import *
import Display
import SpritesClass

pygame.init()

display = Display.Display('games')
screen = display.getScreen()

playerList = SpritesClass.PlayerList()
bulletList = SpritesClass.BulletList()

player1 = SpritesClass.Player(50,250,180,1)
player2 = SpritesClass.Player(450,250,180,2)

playerList.append(player1)
playerList.append(player2)

done = False
end = False

FPSCLOCK = pygame.time.Clock()

def get_next_frame(act):

	if act[7]==1:
		bulletList.append(SpritesClass.Bullet(playerList.getList()[0]))

	pl = playerList.getList()
	pl[0].update(act)

	bulletList.updateAll()

	 # --- Collision Processing
	if bulletList.getList() != 0:
		for bullet1 in bulletList.getList():
			if playerList.getList() != 0:
				for player1 in playerList.getList():
					dx = abs(bullet1.getPosition()[0] - player1.getPosition()[0])
					dy = abs(bullet1.getPosition()[1] - player1.getPosition()[1])
					if dx**2 + dy**2 < (bullet1.getRadius() + player1.getRadius())**2:
						print("Player ", player1.getId(), " got Hit!")
						bulletList.remove(bullet1)
						player1.getHit()
						if player1.isDead():
							done = True
							end = True
							winner = bullet1.getShooter().getId()
						reward[player1.getId() - 1] -= 1.05
						reward[bullet1.getShooter().getId() - 1] += 1

			for bullet2 in bulletList.getList():
				if bullet2 != bullet1:
					dx = abs(bullet1.getPosition()[0] - bullet2.getPosition()[0])
					dy = abs(bullet1.getPosition()[1] - bullet2.getPosition()[1])
					if dx**2 + dy**2 < (bullet1.getRadius() + bullet2.getRadius())**2:
						bulletList.remove(bullet1)
						bulletList.remove(bullet2)
	if playerList.getList() != 0:
		for player1 in playerList.getList():
			for player2 in playerList.getList():
				if player1 != player2:
					dx = abs(player1.getPosition()[0] - player2.getPosition()[0])
					dy = abs(player1.getPosition()[1] - player2.getPosition()[1])
					if dx**2 + dy**2 < (player1.getRadius() + player2.getRadius())**2:
						player1.cancelUpdate()
						player2.cancelUpdate()

	screen.fill((255,255,255))
	display.drawAllPlayers(playerList)
	display.drawAllBullets(bulletList)

	pos0 = player1.getPosition()
	pos1 = player2.getPosition()

	dx = pos1[0]-pos0[0]
	dy = pos1[1]-pos0[1]

	pygame.display.flip()
	FPSCLOCK.tick(1)

	return [dx,dy]