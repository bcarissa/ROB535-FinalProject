import sys
sys.path.append('.')
import numpy as np
from PIL import Image
import random
import pygame
from os import path
from scipy.ndimage import binary_erosion
from utils.Landmark import *

class world2d:

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 24}

    def __init__(self,height=20,width=20,grid_size=1.0):
        # binary_image = Image.open("world/output_binary_100-1.png").convert("L") 
        # binary_matrix = ((np.array(binary_image)/255).astype(int)) # ;print(binary_matrix)
        # height, width = binary_matrix.shape
        # length = max(height,width) 
        length = 10     
        
        # self.height = height
        # self.width = width
        self.grid_size = grid_size

        self.obs_xy = [];self.obs_rowcol = []
        self.rows = length;print("rows w/ bounds:",self.rows)
        self.cols = length;print("cols w/ bounds:",self.cols)
        self.fence_grid = np.zeros((self.rows, self.cols), dtype=int)
        # for r in range(self.rows):
        #     for c in range(self.cols):
        #         if binary_matrix[r,c]==0:
        #             self.add_fence(self.rows-1-r,c)
        # print(self.fence_grid);print()
        # self.fence_grid = self.hollow_obstacles(self.fence_grid)
        # print(self.fence_grid)
        
        # self.start = [int((self.rows-1)/random.uniform(int(self.rows/5), self.rows)),int((self.cols-1)/random.uniform(1, max(int(self.cols/5),1)))]
        self.start = [0,0]
        # self.end = [int((self.rows-1)/random.uniform(int(self.rows/5), self.rows)),int((self.cols-1)/random.uniform(1, max(int(self.cols/5),1)))]
        self.end = [self.rows-1,self.cols-1]
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
        # self.selectStart()
        # self.selectEnd()
        
    def hollow_obstacles(self,map_array):
        eroded_map = binary_erosion(map_array, structure=np.ones((3, 3))).astype(int)
        hollow_map = map_array - eroded_map
        return hollow_map

    def add_fence(self, grid_y, grid_x):
        """Set a fence at the specified grid coordinates."""
        if 0 <= grid_x < self.cols and 0 <= grid_y < self.rows:
            self.fence_grid[grid_y, grid_x] = 1
            # print("adding(x,y):",[grid_x,grid_y])
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
                printout = '#' if self.fence_grid[y][x] else '0'
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

    def selectStart(self):
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
        
        # button area
        for column in range(self.cols):
            self.window_surface.blit(self.select_imgs[3], (column*self.cell_size[0],self.rows*self.cell_size[1])) # label area

        # everyone
        for p in range(self.rows*self.cols):
            row, col = np.unravel_index(p, (self.rows, self.cols))
            pos = (col * self.cell_size[0], row * self.cell_size[1])
            if(self.fence_grid[(self.rows-1)-row][col]==1):
                # print("an obs")
                self.window_surface.blit(self.select_imgs[1], pos)
            else:
                # print("not an obs")
                self.window_surface.blit(self.select_imgs[0], pos)

        color = (0, 0, 0) # black
        smallfont = pygame.font.SysFont('Corbel', 35)
        text = smallfont.render('select start', True, color)
        self.window_surface.blit(text,(self.window_size[0]/2-30,self.window_size[1]-self.cell_size[0]+10))

        notSelected = True
        while notSelected:
            mouse = pygame.mouse.get_pos()
            for ev1 in pygame.event.get():

                if ev1.type == pygame.QUIT:
                    pygame.quit()

                if ev1.type == pygame.MOUSEBUTTONDOWN:
                    col_sel = int(mouse[0] / self.cell_size[0])
                    row_sel = int(mouse[1] / self.cell_size[1])
                    if not [(self.rows-1)-row_sel,col_sel] in self.obs_rowcol:
                        self.start = [(self.rows-1)-row_sel,col_sel]
                        self.window_surface.blit(self.select_imgs[2], (col_sel*self.cell_size[0],row_sel * self.cell_size[1]))
                        print("selected ",self.start)
                        notSelected = False
                    else:
                        print("cannot start on obs!",[(self.rows-1)-row_sel,col_sel])


            pygame.event.pump()
            pygame.display.update()
            clk.tick(self.metadata["render_fps"])
        pygame.quit()

    def selectEnd(self):
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
            if(self.fence_grid[(self.rows-1)-row][col]==1):
                # print("an obs")
                self.window_surface.blit(self.select_imgs[1], pos)
            else:
                # print("not an obs")
                self.window_surface.blit(self.select_imgs[0], pos)

        color = (0, 0, 0) # black
        smallfont = pygame.font.SysFont('Corbel', 35)
        text = smallfont.render('select End', True, color)
        self.window_surface.blit(text,(self.window_size[0]/2-30,self.window_size[1]-self.cell_size[0]+10))

        notSelected = True
        while notSelected:
            mouse = pygame.mouse.get_pos()
            for ev1 in pygame.event.get():

                if ev1.type == pygame.QUIT:
                    pygame.quit()

                if ev1.type == pygame.MOUSEBUTTONDOWN:
                    col_sel = int(mouse[0] / self.cell_size[0])
                    row_sel = int(mouse[1] / self.cell_size[1])
                    if not (([(self.rows-1)-row_sel,col_sel] in self.obs_rowcol) or ([(self.rows-1)-row_sel,col_sel] == self.start)) :
                        self.end = [(self.rows-1)-row_sel,col_sel]
                        self.window_surface.blit(self.select_imgs[2], (col_sel*self.cell_size[0],row_sel * self.cell_size[1]))
                        print("selected ",self.end)
                        notSelected = False
                    else:
                        print("cannot start on obs or start!!",[(self.rows-1)-row_sel,col_sel])


            pygame.event.pump()
            pygame.display.update()
            clk.tick(self.metadata["render_fps"])
        pygame.quit()

def main():

    robot_world = world2d()

    pass

if __name__ == '__main__':
    main()
