import sys
sys.path.append('.')
import yaml
import matplotlib.pyplot as plt


import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped

from system.RobotState import *
from comm.path_publisher import *
from comm.marker_publisher import *
from world.mdp import *
from utils.DataHandler import *
from utils.filter_initialization import filter_initialization
from utils.system_initialization import system_initialization
from utils.utils import *
from mpc.MPC_Controller import *

class RobotSystem:
    

    def __init__(self, world=None):

        rospy.init_node('robot_state_estimator', anonymous=True)

        # load params
        with open("config/settings.yaml", 'r') as stream:
            param = yaml.safe_load(stream)

        # load initial state and mean
        init_state_cov = np.diag(param['initial_state_variance'])**2
        self.loop_sleep_time = param['loop_sleep_time']

        # load world and landmarks
        if world is not None:
            self.world = world
            self.world.print_grids()
        else:
            print("Plase provide a world!")
        
        # MDP --------------------------------------------------------
        self.start = [0,0] # delete this and just state? # y,x
        self.start_xy = [0.5,0.5]
        self.end = [self.world.rows-1,self.world.cols-1] # y,x / row,col
        if self.start in self.world.obs_rowcol:
            self.world.remove_fence(self.start[0],self.start[1])
        if self.end in self.world.obs_rowcol:
            self.world.remove_fence(self.end[0],self.end[1])
        
        self.MDPcore = mdp(self,self.loop_sleep_time)
        self.target_pub = path_publisher()
        self.targetGrid_visualizer = marker_publisher(self.world)
        # MPC --------------------------------------------------------
        self.mpc_pub = path_publisher()
        self.MPC_controller = MPC_Controller(param,self.loop_sleep_time)

        

        
    def plotPath(self):
        x = [self.MDPcore.contiPath[i][0] for i in range(len(self.MDPcore.contiPath))]
        y = [self.MDPcore.contiPath[i][1] for i in range(len(self.MDPcore.contiPath))]
        fig, ax = plt.subplots()
        ax.plot(x, y, marker='o', linestyle='-', color='b')   
        ax.set_title('Continuous Route')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        ax.set_xticks(np.arange(min(x), max(x)+1, 1))  # x-axis grid at intervals of 1
        ax.set_yticks(np.arange(min(y), max(y)+1, 1))
        ax.grid(True)
        plt.show()

    def run_filter(self):
        
        print("running MDP...")
        self.MDPcore.runMDP()
        print("showing route plot...")
        self.plotPath()
        print("plotting grid route...")
        for i in range(5):
            # publish target grid
            self.targetGrid_visualizer.publish_markers()
            rospy.sleep(self.loop_sleep_time)
        # publish target path
        print("publishing continuous target path...")
        for i in range(len(self.MDPcore.contiPath)):
            self.target_pub.publish_target_path(self.MDPcore.contiPath[i])

        print("init MPC...")
        x_bar = np.array(self.MDPcore.mpcTargets)  # 轨迹的状态序列 (每个状态为4维)
        u_bar = np.zeros((len(x_bar), 2))  # 初始化控制输入（假设为2维：加速度和方向盘角度）
        # 假设x0是机器人当前位置
        x_last = [self.start_xy[0], self.start_xy[1], 0, 2]
        print("loop run MPC...")
        for i in range(1000):
            # 使用MPC计算控制输入
            u_act = self.MPC_controller.CMPC_Controller( x_bar, u_bar, x_last)
            # 执行控制输入并更新状态
            x_curr = self.MPC_controller.apply_control(u_act, x_last)
            self.mpc_pub.publish_mpc_path(x_curr[0:2])
            x_last = x_curr
            rospy.sleep(self.loop_sleep_time)

    
    #maybe we can define more specific function to output each step or prediction or correction
        

def main():

    rob_sys = RobotSystem()

    pass

if __name__ == '__main__':
    main()