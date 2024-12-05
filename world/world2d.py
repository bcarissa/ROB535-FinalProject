import sys
sys.path.append('.')
import numpy as np

import pygame
from os import path

from utils.Landmark import *

class world2d:

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 24}

    def __init__(self,height=10,width=10,grid_size=1.0):
        # self.height = height
        # self.width = width
        self.grid_size = grid_size

        self.rows = height
        print("rows w/ bounds:",self.rows)
        self.cols = width
        print("cols w/ bounds:",self.cols)
        self.fence_grid = np.zeros((self.rows, self.cols), dtype=int)
        self.obs_xy = []
        self.obs_rowcol = []
        self.path = []

        self.cell_size = (60, 60)
        self.window_size = (
            (self.cols) * self.cell_size[1], # x
            (self.rows+1) * self.cell_size[0], # y
        )
        self.window_surface = None
        self.clock = None
        self.obs_imgs = None
        self.select_imgs = None

        self.selectObs()

    def add_fence(self, grid_y, grid_x):
        """Set a fence at the specified grid coordinates."""
        if 0 <= grid_x < self.cols and 0 <= grid_y < self.rows:
            self.fence_grid[grid_y, grid_x] = 1
            print("adding(x,y):",[grid_x,grid_y])
            self.obs_xy.append([grid_x,grid_y])
            self.obs_rowcol.append([grid_y,grid_x])

    def remove_fence(self, grid_y, grid_x):
        """Remove a fence at the specified grid coordinates."""
        if 0 <= grid_x < self.cols and 0 <= grid_y < self.rows:
            self.fence_grid[grid_y, grid_x] = 0
            print("removing(x,y):",[grid_x,grid_y])
            self.obs_xy.remove([grid_x,grid_y])
            self.obs_rowcol.remove([grid_y,grid_x])


    def add_mdp_path(self, grid_y, grid_x):
        """Set a MDP path at the specified grid coordinates."""
        if 0 <= grid_x < self.cols and 0 <= grid_y < self.rows:
            self.fence_grid[grid_y, grid_x] = 2
            print("MDP path(x,y):",[grid_x,grid_y])
    
    def print_grids(self):
        for y in range(-1,-(self.rows)-1,-1):
            printLine = ''
            for x in range(self.cols):
                printout = '#' if self.fence_grid[y,x] else '0'
                printLine+=printout
            print(printLine)

    def selectObs(self):
        clk = pygame.time.Clock()
        pygame.init()
        pygame.display.init()
        pygame.display.set_caption("select")
        self.window_surface = pygame.display.set_mode(self.window_size)
        if self.select_imgs is None:
            select = [
                path.join(path.dirname(__file__), "img/unselected.png"),
                path.join(path.dirname(__file__), "img/selected.png"),
                path.join(path.dirname(__file__), "img/selected_un.png"),
                path.join(path.dirname(__file__), "img/none.png"),
            ]
            self.select_imgs = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size) for f_name in select
            ]
        
        for column in range(self.cols):
            self.window_surface.blit(self.select_imgs[3], (column*self.cell_size[0],self.rows*self.cell_size[1])) # label area

        for p in range(self.rows*self.cols):
            row, col = np.unravel_index(p, (self.rows, self.cols))
            pos = (col * self.cell_size[0], row * self.cell_size[1])
            self.window_surface.blit(self.select_imgs[0], pos)

        color = (0, 0, 0) # black
        smallfont = pygame.font.SysFont('Corbel', 35)
        text = smallfont.render('select', True, color)
        self.window_surface.blit(text,(self.window_size[0]/2-30,self.window_size[1]-self.cell_size[0]+10))

        flag=True
        while flag:
            mouse = pygame.mouse.get_pos()
            for ev1 in pygame.event.get():

                if ev1.type == pygame.QUIT:
                    pygame.quit()

                if ev1.type == pygame.MOUSEBUTTONDOWN:
                    col_sel = int(mouse[0] / self.cell_size[0])
                    row_sel = int(mouse[1] / self.cell_size[1])
                    if row_sel == self.rows: # exceed maximum
                        print("finished")
                        flag = False
                    elif not [(self.rows-1)-row_sel,col_sel] in self.obs_rowcol:
                        self.window_surface.blit(self.select_imgs[1], (col_sel*self.cell_size[0],row_sel * self.cell_size[1]))
                        self.add_fence((self.rows-1)-row_sel,col_sel)
                    else:
                        self.window_surface.blit(self.select_imgs[0], (col_sel*self.cell_size[0],row_sel * self.cell_size[1]))
                        self.remove_fence((self.rows-1)-row_sel,col_sel)


            pygame.event.pump()
            pygame.display.update()
            clk.tick(self.metadata["render_fps"])
        pygame.quit()

def main():

    robot_world = world2d()

    pass

if __name__ == '__main__':
    main()
