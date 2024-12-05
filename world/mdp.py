from gym import Env,spaces
import numpy as np
import mdptoolbox

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=1000)

costMove = 1
costScan = 4
costBump = 5

class mdp(Env):
    def __init__(self,robot):
        self.robot = robot
        self.col = self.robot.world.cols
        self.row = self.robot.world.rows
        self.obstacles = self.robot.world.obs_rowcol # row,col
        self.start = self.robot.start
        self.end = self.robot.end

        self.MDP_state = [[self.start[0],self.start[1]],False] # row,col,bump

        self.action = {
                             'LEFT': [0, -1],
                             'RIGHT': [0, 1],
                             'FORWARD': [1, 0],
                             'BACKWARD': [-1, 0],
                             }
        self.observation_space = spaces.Discrete(self.col*self.row)
        self.timeLengthMax = self.row*self.col
        self.g0 = 0
        self.contiPath = []

    def setStateReward(self,R,point:list,reward): # reward when stepping onto this point
        if (0 <= point[0]+1 < self.row and 0 <= point[1] < self.col):
            R[(point[0]+1)*self.row+point[1],3] = reward # plus or minus????????????????????????????????????????????????
        if (0 <= point[0]-1 < self.row and 0 <= point[1] < self.col):
            R[(point[0]-1)*self.row+point[1],2] = reward
        if (0 <= point[0] < self.row and 0 <= point[1]-1 < self.col):
            R[point[0]*self.row+point[1]-1,1] = reward
        if (0 <= point[0] < self.row and 0 <= point[1]+1 < self.col):
            R[point[0]*self.row+point[1]+1,0] = reward
        return R

    def boundaryVoidReward(self,R,point:list,reward):
        if not (0 <= point[0]+1 < self.row and 0 <= point[1] < self.col):
            R[point[0]*self.row+point[1],2] = reward
        if not (0 <= point[0]-1 < self.row and 0 <= point[1] < self.col):
            R[point[0]*self.row+point[1],3] = reward
        if not (0 <= point[0] < self.row and 0 <= point[1]-1 < self.col):
            R[point[0]*self.row+point[1],0] = reward
        if not (0 <= point[0] < self.row and 0 <= point[1]+1 < self.col):
            R[point[0]*self.row+point[1],1] = reward
        return R

    def blockTransition(self,P,point:list,S):
        if (0 <= point[0]+1 < self.row and 0 <= point[1] < self.col):
            P[3,(point[0]+1)*self.row+point[1],:]=[1 if x==(point[0]+1)*self.row+point[1] else 0 for x in range(S)]
        if (0 <= point[0]-1 < self.row and 0 <= point[1] < self.col):
            P[2,(point[0]-1)*self.row+point[1],:]=[1 if x==(point[0]-1)*self.row+point[1] else 0 for x in range(S)]
        if (0 <= point[0] < self.row and 0 <= point[1]-1 < self.col):
            P[1,point[0]*self.row+point[1]-1,:]=[1 if x==point[0]*self.row+point[1]-1 else 0 for x in range(S)]
        if (0 <= point[0] < self.row and 0 <= point[1]+1 < self.col):
            P[0,point[0]*self.row+point[1]+1,:]=[1 if x==point[0]*self.row+point[1]+1 else 0 for x in range(S)]
        return P

    def endTransition(self,P,end:list,S):
        if (0 <= end[0]+1 < self.row and 0 <= end[1] < self.col):
            P[2,end[0]*self.row+end[1],:]=[1 if x==end[0]*self.row+end[1] else 0 for x in range(S)]
        if (0 <= end[0]-1 < self.row and 0 <= end[1] < self.col):
            P[3,end[0]*self.row+end[1],:]=[1 if x==end[0]*self.row+end[1] else 0 for x in range(S)]
        if (0 <= end[0] < self.row and 0 <= end[1]-1 < self.col):
            P[0,end[0]*self.row+end[1],:]=[1 if x==end[0]*self.row+end[1] else 0 for x in range(S)]
        if (0 <= end[0] < self.row and 0 <= end[1]+1 < self.col):
            P[1,end[0]*self.row+end[1],:]=[1 if x==end[0]*self.row+end[1] else 0 for x in range(S)]
        return P

    def selectMovement(self,policy:list,localStart:list):
            p = policy[localStart[0]*self.row+localStart[1]]
            action = [x for x in self.action.keys()]
            nextMove = action[p]
            return nextMove

    def valueIteration(self, r1=-1, r2=10, r3=-10): 
        r2 = self.row*self.col
        r3 = -r2
        S_2D = self.row * self.col
        assert (r1, r3 <= 0) and (r2 > 0), "r1, r3 are costs, r2 is reward."
        
        """-------------------P-------------------"""
        P = np.zeros((4, S_2D, S_2D))
        # blocks are taken care of after this
        
        for x in range(0,S_2D):
            if x % self.row == 0:
                P[0,x,x]=1
            else:
                P[0, x, x - 1] = 1

        # right
        for x in range(0, S_2D):
            if (x + 1) % self.row == 0:
                P[1, x, x] = 1
            else:
                P[1, x, x + 1] = 1

        # forward
        for x in range(0, S_2D - self.row):
            P[2, x, x + self.row] = 1
        for x in range(S_2D - self.row, S_2D):
            P[2, x, x] = 1

        # backward
        for x in range(self.row, S_2D):
            P[3, x, x - self.row] = 1
        for x in range(0, self.row):
            P[3, x, x] = 1

        for point in self.obstacles:
            P = self.blockTransition(P,point,S_2D)

        #no action for end state
        P = self.endTransition(P,self.end,S_2D)
        #P = self.blockTransition(P,self.start,S_2D)


        """-------------------R-------------------"""
        # set for normal
        R = r1 * np.ones((S_2D, 4))

        # set for end
        R = self.setStateReward(R,self.end,r2)

        # set for boundary
        for r in range(self.row):
            for c in range(self.col):
                R = self.boundaryVoidReward(R,[r,c],r3)

        # set for obstacles
        for pp in self.obstacles:
            R = self.setStateReward(R, pp, r3)
        """print area"""
        # print("curr R")
        # print(R)
        # print("curr P")
        # print(P)
        """print area"""
        return (P, R)
    
        

    def step(self):
        g_curr = 0
        self.MDP_state[1] = False

        """Value iteration"""
        P, R = self.valueIteration()
        vi = mdptoolbox.mdp.ValueIteration(P, R, 0.99, 0.1)
        vi.run()
        policy = list(vi.policy)
        """Value iteration"""

        nextMove = self.selectMovement(policy,self.MDP_state[0])
        print("next move is {}".format(nextMove))

        """print policy"""
        # print("***Policy table***")
        # for y in range(self.row):
        #     for x in range(self.col):
        #         print(" {} ".format(policy[x + self.row * y]), end='')
        #         # print(vi.policy[1]<5)
        #     print()
        # print("***Policy table***")
        # print()
        """print policy"""

        
        g_curr = costMove
        temp = self.MDP_state[0]
        self.MDP_state[0] = np.add(self.MDP_state[0][:3], self.action['{}'.format(nextMove)])
        self.MDP_state[0] = self.MDP_state[0].tolist()
        if self.MDP_state[0] in self.obstacles:
            self.MDP_state[1] = True
            self.MDP_state[0] = temp # stuck, so revert
            g_curr = costBump
        self.g0 += g_curr

        # self.pastPath.append(self.MDP_state[0])
        self.robot.world.path.append(self.MDP_state[0])
        self.robot.world.add_mdp_path(self.MDP_state[0][0],self.MDP_state[0][1])

        self.timeLengthMax -= 1

        done = True if (self.timeLengthMax <= 0 or self.MDP_state[0]==self.end) else False
        info = {}
        return self.MDP_state, self.g0, done, info



    def reset(self):
        self.MDP_state[0] = [self.start[0], self.start[1]]
        self.MDP_state[1] = False
        self.g0 = 0
        self.contiPath = []
        self.robot.world.path = []
        self.timeLengthMax = self.row*self.col

    def render(self, mode="human"):
        print("position (row col): {}".format(self.MDP_state[0]))
        if self.MDP_state[1] == True:
            print("Bump!")

        for j in range(self.row-1,-1,-1):
            for k in range(self.col):
                if self.MDP_state[0][0]==j and self.MDP_state[0][1]==k: #robot
                    if self.MDP_state[1] == False: #redundant!!!
                        print(" r ",end='')
                    else:
                        print(" x ",end='')
                elif [j,k] in self.obstacles:
                    print(" = ",end='')
                elif [j,k] == self.end or [j,k] == self.start: #end
                    print(" □ ",end='')
                else: #empty
                    print(" o ",end='')
            print()
        print()

    def isTurn(self,idx):
        # two points
        grid_size = self.robot.world.grid_size
        if idx-1>=0:
            point_last = self.robot.world.path[idx-1]
            x_last = point_last[1] * grid_size + grid_size / 2
            y_last = point_last[0] * grid_size + grid_size / 2
        else:
            point_last = self.robot.world.path[idx]
            x_last = point_last[1] * grid_size + grid_size / 2
            y_last = point_last[0] * grid_size + grid_size / 2

        point_curr = self.robot.world.path[idx]
        x_curr = point_curr[1] * grid_size + grid_size / 2
        y_curr = point_curr[0] * grid_size + grid_size / 2

        if idx+1<len(self.robot.world.path):
            point_nxt = self.robot.world.path[idx+1]
            x_nxt = point_nxt[1] * grid_size + grid_size / 2
            y_nxt = point_nxt[0] * grid_size + grid_size / 2
        else:
            point_nxt = None
            x_nxt = x_curr
            y_nxt = y_curr

        # curr point, then points within, NO next point
        if idx-1>=0 and idx+1<len(self.robot.world.path) and ((y_curr - y_last != y_nxt - y_curr) or (x_curr - x_last != x_nxt - x_curr)):
            print("turn!",idx)
            return True,x_last,y_last,point_last,x_curr,y_curr,point_curr,x_nxt,y_nxt,point_nxt
        return False,x_last,y_last,point_last,x_curr,y_curr,point_curr,x_nxt,y_nxt,point_nxt

    def generateContinuousPath(self):
        grid_size = self.robot.world.grid_size
        # generated = [False for i in range(len(self.robot.world.path))]
        startPoint = self.robot.world.path[0]
        lastEnd = [startPoint[0] * grid_size + grid_size / 2,startPoint[1] * grid_size + grid_size / 2]
        idx = 0
        while idx<len(self.robot.world.path)-1:
            turn,x_last,y_last,point_last,x_curr,y_curr,point_curr,x_nxt,y_nxt,point_nxt = self.isTurn(idx)
            if not turn:
                currEnd = [x_curr+(x_nxt - x_curr)/2.0,y_curr+(y_nxt - y_curr)/2.0]
                distance = np.sqrt((currEnd[0] - lastEnd[0])**2 + (currEnd[1] - lastEnd[1])**2)
                num_points = int(np.ceil(100 * distance)) # 100 point per unit
                x_values = np.linspace(lastEnd[0], currEnd[0], num_points, endpoint=False)
                y_values = np.linspace(lastEnd[1], currEnd[1], num_points, endpoint=False)
                lastEnd = currEnd
            else:
                # turn part
                num_points = 100
                angles = np.linspace(0,0.5*np.pi,num_points,endpoint=False)
                x_sign = np.sign (np.sign(point_curr[1]-point_last[1]) + np.sign(point_nxt[1] - point_curr[1]) )
                y_sign = np.sign (np.sign(point_curr[0]-point_last[0]) + np.sign(point_nxt[0] - point_curr[0]) )
                turnUpDown = True if (x_curr-x_last) and (y_nxt - y_curr) else False
                print("updown:",turnUpDown)                
                turn_start = [x_last+(x_curr - x_last)/2.0,y_last+(y_curr - y_last)/2.0]
                print(turn_start,x_sign,y_sign)
                x_values = turn_start[0]+x_sign*np.sin(angles)*0.5 if turnUpDown else turn_start[0]+x_sign*(1-np.cos(angles))*0.5
                y_values = turn_start[1]+y_sign*(1-np.cos(angles))*0.5 if turnUpDown else turn_start[1]+y_sign*np.sin(angles)*0.5
                lastEnd = [lastEnd[0]+(x_nxt - x_last)/2.0,lastEnd[1]+(y_nxt - y_last)/2.0]

                

            idx = idx+1
            for i in range(len(x_values)):
                self.contiPath.append([x_values[i], y_values[i]])





    def runMDP(self):
        g0 = 0
        self.reset()
        print("-----------MDP-----------")
        print("start position:{}".format(self.start))
        steptime = 0

        self.robot.world.path.append(self.start)
        self.robot.world.add_mdp_path(self.start[0],self.start[1])
        while True:
            print("step:{}----".format(steptime))
            self.MDP_state,g0,done,info = self.step()
            # self.render()
            steptime+=1
            if done:
                break
        print('cost:{}'.format(str(self.g0)))
        print("-----------MDP-----------")
        print()
        self.close()
        print("MDP path obtained w/",str(len(self.robot.world.path)),"points:",self.robot.world.path)
        self.generateContinuousPath() 



