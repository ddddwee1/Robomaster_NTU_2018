import pygame
from pygame.locals import *
import random, math, sys
import Display
import Functions
import SpritesClass
import InputHandling
from socket import *
import time, random
import datetime

"""
Prepare the simulation of the game in the server
"""
# --- Create the window
# Initialize Pygame
pygame.init()

# Set the constraint of the environment
arenaWidth = 500 # The width of the arena in [cm]
arenaHeight = 800 # The height of the arena in [cm]
scale = 1 # The scale of the display to the real size in [cm/px]
framerate = 30 # The frame rate in [frame/s]

# Setup the screen
display = Display.Display(scale, arenaWidth, arenaHeight)
screen = display.getScreen()

# --- Sprite lists
# List of each player
playerList = SpritesClass.PlayerList()

# List of each bullet
bulletList = SpritesClass.BulletList()

# List of each bullet
obstacleList = SpritesClass.ObstacleList()

# --- Create the sprites
# Team 1
player1 = SpritesClass.Player(1, 110,50, framerate, scale, 2)
player2 = SpritesClass.Player(2, 50,110, framerate, scale, 1)
# Team 2
player3 = SpritesClass.Player(3, 390,750, framerate, scale, 4)
player4 = SpritesClass.Player(4, 450,690, framerate, scale, 3)

playerList.append(player1)
playerList.append(player2)
playerList.append(player3)
playerList.append(player4)

# Obstacle list
obstacle = SpritesClass.RectangleObstacle((150,180),(270,210))
obstacleList.append(obstacle)
obstacle = SpritesClass.RectangleObstacle((370,120),(400,200))
obstacleList.append(obstacle)
obstacle = SpritesClass.RectangleObstacle((220,0),(250,80))
obstacleList.append(obstacle)
obstacle = SpritesClass.RectangleObstacle((0,310),(200,340))
obstacleList.append(obstacle)
obstacle = SpritesClass.RectangleObstacle((300,460),(500,490))
obstacleList.append(obstacle)
obstacle = SpritesClass.RectangleObstacle((230,590),(350,620))
obstacleList.append(obstacle)
obstacle = SpritesClass.RectangleObstacle((100,600),(130,680))
obstacleList.append(obstacle)
obstacle = SpritesClass.RectangleObstacle((250,720),(280,800))
obstacleList.append(obstacle)
# Loop until the user clicks the close button.
done = False
end = False

# Used to manage how fast the screen updates
clock = pygame.time.Clock()

# -------- Main Program Loop -----------
while not done:

    # Reward for training AI purposes
    reward = [0.05,0.05,0.05,0.05]

    # The action Tuple goes here
    # actP1, actP2. actP3, actP4

    # TODO: Comment this part. This is only for debugging
    actP1, status = InputHandling.checkInput()


    # --- Game logic
    # Create bullet if the player1/player2 shoots
    if actP1[7] == 1:
        bulletList.append(SpritesClass.Bullet(playerList.getList()[0]))
        reward[0] -= 0.1

    # TODO: Uncomment this!!!
    """
    if actP2[7] == 1:
        bulletList.append(SpritesClass.Bullet(playerList.getList()[1]))
        reward[1] -= 0.1

    if actP3[7] == 1:
        bulletList.append(SpritesClass.Bullet(playerList.getList()[2]))
        reward[2] -= 0.1

    if actP4[7] == 1:
        bulletList.append(SpritesClass.Bullet(playerList.getList()[3]))
        reward[3] -= 0.1
    """

    # Call the update() method on all the existing sprites
    # TODO: Uncomment this
    playerList.getList()[0].update(actP1, obstacleList, arenaWidth, arenaHeight)
    #playerList.getList()[1].update(actP2, obstacleList, arenaWidth, arenaHeight)
    #playerList.getList()[2].update(actP3, obstacleList, arenaWidth, arenaHeight)
    #playerList.getList()[3].update(actP4, obstacleList, arenaWidth, arenaHeight)

    bulletList.updateAll(obstacleList, arenaWidth, arenaHeight)

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
                            playerList.killed(player1)
                        # TODO: Implement penalty for hitting robot of the same team?
                        # Please check this. Thank you
                        if player1.getId() == bullet1.getShooter().getTeamMates():
                            reward[player1.getId() - 1] -= 0.05
                            reward[bullet1.getShooter().getId() - 1] -= 2
                        else:
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

    # Check if all players on a team is dead. If yes, end the simulation
    # TODO: Is this correct? : There are two robots on a Team
    if playerList.getKilled != 0:
        if playerList.getKilled()[0] == 1 and playerList.getKilled()[1] == 1:
            end = True
        if playerList.getKilled()[2] == 1 and playerList.getKilled()[3] == 1:
            end = True


    # Update the screen
    screen.fill((255,255,255))
    display.drawAllPlayers(playerList)
    display.drawAllBullets(bulletList)
    display.drawAllObstacles(obstacleList)

    # Go ahead and update the screen with what is drawn.
    pygame.display.flip()

    # Output the current frame data
    # TODO: Output how to check if the enemy is in sight or not
    #       The field of view of the player is 70 degrees
    #       The field of view must also be obstructed by the obstacle

    if end == True:
        break;

    # --- Limit to 30 frames per second
    clock.tick(framerate)

if end == True:
    print("Game has ended.")
pygame.quit()
