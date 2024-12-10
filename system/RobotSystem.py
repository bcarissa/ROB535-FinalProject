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
from mpc.cmpc_utils import *

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
            # self.world.print_grids()
        else:
            print("Plase provide a world!")
        
        # MDP --------------------------------------------------------
        self.start = self.world.start # delete this and just state? # y,x
        # self.start_xy = [0,0]
        # self.end = [self.world.rows-1,self.world.cols-1] # y,x / row,col
        self.end = self.world.end

        if self.start in self.world.obs_rowcol:
            self.world.remove_fence(self.start[0],self.start[1])
        if self.end in self.world.obs_rowcol:
            self.world.remove_fence(self.end[0],self.end[1])
        
        self.MDPcore = mdp(self,self.loop_sleep_time)
        self.target_pub = path_publisher()
        self.targetGrid_visualizer = marker_publisher(self.world)
        # MPC --------------------------------------------------------
        self.mpc_result = []
        self.x_last = None
        self.mpc_pub = path_publisher()
        self.MPC_controller = MPC_Controller(param,self.loop_sleep_time)

        

        
    def plotPath(self):
        # x = [self.MDPcore.mpcTargets[i][0] for i in range(len(self.MDPcore.mpcTargets))]
        # y = [self.MDPcore.mpcTargets[i][1] for i in range(len(self.MDPcore.mpcTargets))]
        # yaw = [self.MDPcore.mpcTargets[i][2] for i in range(len(self.MDPcore.mpcTargets))]
        # t = [0.1*i for i in range(len(self.MDPcore.mpcTargets))]
        x = [self.MDPcore.contiPath[i][0] for i in range(len(self.MDPcore.contiPath))]
        y = [self.MDPcore.contiPath[i][1] for i in range(len(self.MDPcore.contiPath))]
        # yaw = [self.MDPcore.contiPath[i][2] for i in range(len(self.MDPcore.contiPath))]
        t = [0.1*i for i in range(len(self.MDPcore.contiPath))]
        
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        axes = axes.flatten()
        ax = axes[0]
        ax.plot(x, y, marker='o', linestyle='-', color='b')   
        ax.set_title('Continuous Route')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        ax.set_xticks(np.arange(min(x), max(x)+1, 1))  # x-axis grid at intervals of 1
        ax.set_yticks(np.arange(min(y), max(y)+1, 1))
        ax.grid(True)

        # ax = axes[1]
        # ax.plot(t,yaw, linestyle='-', color='r')   
        # ax.set_title('yaw states')
        # # ax.set_xlabel('X')
        # ax.set_ylabel('yaw angle')
        # ax.grid(True)
        # plt.show()

    def plotMDPPath(self):
        x = [self.MDPcore.mpcTargets[i][0] for i in range(len(self.MDPcore.mpcTargets))]
        y = [self.MDPcore.mpcTargets[i][1] for i in range(len(self.MDPcore.mpcTargets))]
        yaw = [self.MDPcore.mpcTargets[i][2] for i in range(len(self.MDPcore.mpcTargets))]
        t = [0.1*i for i in range(len(self.MDPcore.mpcTargets))]
        # x = [self.MDPcore.contiPath[i][0] for i in range(len(self.MDPcore.contiPath))]
        # y = [self.MDPcore.contiPath[i][1] for i in range(len(self.MDPcore.contiPath))]
        # yaw = [self.MDPcore.contiPath[i][2] for i in range(len(self.MDPcore.contiPath))]
        # t = [0.1*i for i in range(len(self.MDPcore.contiPath))]
        
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        axes = axes.flatten()
        ax = axes[0]
        ax.plot(x, y, marker='o', linestyle='-', color='b')   
        ax.set_title('Continuous Route')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        ax.set_xticks(np.arange(min(x), max(x)+1, 1))  # x-axis grid at intervals of 1
        ax.set_yticks(np.arange(min(y), max(y)+1, 1))
        ax.grid(True)

        ax = axes[1]
        ax.plot(t,yaw, linestyle='-', color='r')   
        ax.set_title('yaw states')
        # ax.set_xlabel('X')
        ax.set_ylabel('yaw angle')
        ax.grid(True)
        plt.show()
    
    def plotTargetResult(self):
        x = [self.MDPcore.mpcTargets[i][0] for i in range(len(self.MDPcore.mpcTargets))]
        y = [self.MDPcore.mpcTargets[i][1] for i in range(len(self.MDPcore.mpcTargets))]
        yaw = [self.MDPcore.mpcTargets[i][2] for i in range(len(self.MDPcore.mpcTargets))]
        t = [0.1*i for i in range(len(self.MDPcore.mpcTargets))]

        x_res = [self.mpc_result[i][0] for i in range(len(self.mpc_result))]
        y_res = [self.mpc_result[i][1] for i in range(len(self.mpc_result))]
        yaw_res = [self.mpc_result[i][2] for i in range(len(self.mpc_result))]
        t_res = [0.1*i for i in range(len(self.mpc_result))]


        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        axes = axes.flatten()
        ax = axes[0]
        ax.plot(x, y, linestyle='-', color='b')   
        ax.plot(x_res, y_res, linestyle='--', color='r')   
        ax.set_title('Continuous Route')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        ax.legend()
        ax.set_xticks(np.arange(min(x), max(x)+1, 1))  # x-axis grid at intervals of 1
        ax.set_yticks(np.arange(min(y), max(y)+1, 1))
        ax.grid(True)

        ax = axes[1]
        ax.plot(t[0:len(t_res)],yaw[0:len(t_res)], linestyle='-', color='b')
        ax.plot(t_res,yaw_res, linestyle='--', color='r')
        ax.set_title('yaw states')
        # ax.set_xlabel('X')
        ax.set_ylabel('yaw angle')
        ax.set_aspect('auto')
        ax.legend()
        ax.grid(True)
        plt.show()

    def run_filter(self):
        
        print("running MDP...")
        self.MDPcore.runMDP()
        print("showing route plot...")
        # self.plotPath()
        self.plotMDPPath()
        print("plotting grid route...")
        
        # publish target grid
        for i in range(4):
            self.targetGrid_visualizer.publish_markers()
            rospy.sleep(self.loop_sleep_time)
        
        # publish target path
        print("publishing continuous target path...")
        for i in range(len(self.MDPcore.contiPath)):
            self.target_pub.publish_target_path(self.MDPcore.contiPath[i])
            # rospy.sleep(self.loop_sleep_time)

        # print("init MPC...")
        # x_bar = np.array(self.MDPcore.mpcTargets)  # 轨迹的状态序列 (每个状态为4维)
        # u_bar = np.zeros((len(x_bar), 2))  # 初始化控制输入（假设为2维：加速度和方向盘角度）
        # 假设x0是机器人当前位置
        # self.x_last = [self.MDPcore.mpcTargets[0][0], self.MDPcore.mpcTargets[0][1], self.MDPcore.mpcTargets[0][2], 0]
        # print("loop run MPC...")
        # for i in range(10):
            # print("time ",0.1*i)
            # 使用MPC计算控制输入
            # u_act = self.MPC_controller.CMPC_Controller( x_bar, u_bar, self.x_last)
            # 执行控制输入并更新状态
            # x_curr = self.MPC_controller.apply_control(u_act, self.x_last)
            # self.mpc_result.append(x_curr)
            # self.mpc_pub.publish_mpc_path(x_curr[0:2])
            # self.x_last = x_curr
            # rospy.sleep(self.loop_sleep_time)
        
        # self.plotTargetResult()

    
    def plot_res(self):
        param = {
            "h":0.05,
            "T":9.5,
            "L_f": 1.,
            "L_r":1.,
            "a_lim": [-10.0, 10.],
            "delta_lim": [-1.5, 1.5]
        }
        Sim = BicycleSim(param)
        u_bar, x_bar = Sim.GenRef()
        T = len(x_bar)*0.05
        param = {
            "h":0.05,
            "T":T,
            "L_f": 1.,
            "L_r":1.,
            "a_lim": [-10.0, 10.],
            "delta_lim": [-1.5, 1.5],
            "x_obs": 49.5,
            "y_obs": 59.5,
        }
        Sim = BicycleSim(param)
        u_bar, x_bar = Sim.GenRef()

        preview = 20
        x0 = np.array(x_bar[0]) # [x, y, theta, v], x 0-5, y 0-5, theta 0-0.6*pi, v 0-5
        controller = lambda x, u, xint: self.MPC_controller.CMPC_Controller(x, u, xint)
        x_log, u_log = Sim.SimVehicle(x_bar, u_bar, preview, x0, controller)
        np.savetxt('system/x_log.txt', x_log, fmt='%.6f', delimiter=',', header='x, y, theta, v', comments='')
        np.savetxt('system/u_log.txt', u_log, fmt='%.6f', delimiter=',', header='acceleration, steering_angle', comments='')

        data_des = np.loadtxt('system/path.txt', delimiter=',',skiprows=1)
        x_des = data_des[:, 0]
        y_des = data_des[:, 1]
        yaw_des = data_des[:, 2]
        v_des = data_des[:, 3]

        ctrl_des = np.loadtxt('system/controls.txt',delimiter=',', skiprows=1)
        a_des = ctrl_des[:,0]
        delta_des = ctrl_des[:,1]

        data_res = np.loadtxt('system/x_log.txt',delimiter=',', skiprows=1)
        x_res = data_res[:, 0]
        y_res = data_res[:, 1]
        yaw_res = data_res[:, 2]
        v_res = data_res[:, 3]

        ctrl_res = np.loadtxt('system/u_log.txt',delimiter=',', skiprows=1)
        a_res = ctrl_res[:,0]
        delta_res = ctrl_res[:,1]


        plt.figure(figsize=(5, 5))
        # plt.plot(x_des, y_des, 'ko-')
        # plt.plot(x_res, x_res, 'ro-')

        plt.plot(x_des, y_des, 'k-', label = "reference")
        plt.plot(x_res, y_res, 'r--', label = "vehicle")
        plt.axis('equal')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('trajectory')
        plt.legend()
        plt.grid()
        plt.savefig('system/trajectory.png')
        plt.show()

        plt.figure(figsize=(20, 3))
        plt.plot(yaw_des, 'k-', label = "reference")
        plt.plot(yaw_res, 'r--', label = "vehicle")
        plt.title('Orientation')
        plt.legend()
        plt.grid()
        plt.savefig('system/Orientation.png')
        plt.show()


        plt.figure(figsize=(20, 3))
        plt.plot(v_des, 'k-', label = "reference")
        plt.plot(v_res, 'r--', label = "vehicle")
        plt.title('Velocity')
        plt.legend()
        plt.grid()
        plt.savefig('system/Velocity.png')
        plt.show()


        plt.figure(figsize=(20, 3))
        plt.plot(a_des, 'k-', label = "reference")
        plt.plot(a_res, 'r--', label = "vehicle")
        plt.title('Acceleration')
        plt.legend()
        plt.grid()
        plt.savefig('system/Acceleration.png')
        plt.show()


        plt.figure(figsize=(20, 3))
        plt.plot(delta_des, 'k-', label = "reference")
        plt.plot(delta_res, 'r--', label = "vehicle")
        plt.title('Steering Angle')
        plt.legend()
        plt.grid()
        plt.savefig('system/Steering.png')
        plt.show()

        for i in range(len(x_res)):
            self.mpc_pub.publish_mpc_path([x_res[i],y_res[i]])

        
        

def main():

    rob_sys = RobotSystem()

    pass

if __name__ == '__main__':
    main()