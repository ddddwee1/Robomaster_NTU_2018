import pygame

def checkInput():
    act = [1,0,0,0,0,0,0,0]
    status = 0
    keyState = pygame.key.get_pressed()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            status = -1

        elif event.type == pygame.KEYDOWN:
            if event.key == 119: act[1] = 1     # W
            if event.key == 97: act[3] = 1      # S
            if event.key == 115: act[2] = 1     # A
            if event.key == 100: act[4] = 1     # D
            if event.key == 275: act[6] = 1     # Right
            if event.key == 276: act[5] = 1     # Left
            if event.key == 273:                # Up
                act[7] = 1
            if event.key == 27: status = -1     # ESC

    if keyState[119]: act[1] = 1
    if keyState[97]: act[3] = 1
    if keyState[115]: act[2] = 1
    if keyState[100]: act[4] = 1
    if keyState[275]: act[6] = 1
    if keyState[276]: act[5] = 1
    # if keyState[273]: act[7] = 1 <- Must not allow bursting shoot
    return act, status
