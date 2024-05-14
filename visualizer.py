import pygame as pg
import torch
from main import NeRFModel, device

# Constants
WIDTH, HEIGHT = 400, 400
BLACK = pg.Color((0,0,0))

pg.init()

w = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("NeRF visualizer")

# Object instances
model = NeRFModel().to(device)
model.load_state_dict(torch.load("model.pth"))

def main():
    # Frame rate
    clock = pg.time.Clock()
    clock.tick(120)

    positions = torch.tensor([[[WIDTH // 2 - j, HEIGHT // 2 - i]
                               for j in range(0, WIDTH)]
                               for i in range(0, HEIGHT)])

    currentFrame = torch.zeros((WIDTH, HEIGHT))

    inApp = True
    while inApp:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                inApp = False
        
        w.fill(BLACK)
        pg.display.update()

if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        quit()