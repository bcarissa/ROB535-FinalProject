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

class RobotSystem:
    

    def __init__(self, world=None):

        rospy.init_node('robot_state_estimator', anonymous=True)

        # load params
        with open("config/settings.yaml", 'r') as stream:
            param = yaml.safe_load(stream)

        # load motion noise and sensor noise
        alphas = np.array(param['alphas_sqrt'])**2
        beta = np.deg2rad(param['beta'])

        # load initial state and mean
        init_state_cov = np.diag(param['initial_state_variance'])**2

        self.system_ = system_initialization(alphas, beta)

        self.filter_name = param['filter_name']

        # load world and landmarks
        if world is not None:
            self.world = world
            self.world.print_grids()
        else:
            print("Plase provide a world!")
        
        # MDP --------------------------------------------------------
        self.start = [0,0] # delete this and just state? # y,x
        self.end = [self.world.rows-1,self.world.cols-1] # y,x / row,col
        print("end is:",self.end)
        if self.start in self.world.obs_rowcol:
            self.world.remove_fence(self.start[0],self.start[1])
        if self.end in self.world.obs_rowcol:
            self.world.remove_fence(self.end[0],self.end[1])
        
        self.MDPcore = mdp(self)
        # MDP --------------------------------------------------------

        # if self.filter_name is not None:
        #     print("Initializing", self.filter_name)
        #     self.filter_ = filter_initialization(self.system_, self.start, init_state_cov, self.filter_name)
        #     self.state_ = self.filter_.getState()
        # else:
        #     print("Please specify a filter name!")
        
        # load data.
        # in real-world application this should be a subscriber that subscribes to sensor topics
        # but for this homework example we load all data at once for simplicity
        
        # self.data_handler = DataHandler()
        # self.data = self.data_handler.load_2d_data()

        # self.num_step = np.shape(self.data['motionCommand'])[0]

        # self.pub = path_publisher()     # filter pose
        # self.cmd_pub = path_publisher() # theoratical command path
        # self.gt_pub = path_publisher()  # actual robot path

        self.target_pub = path_publisher()
        self.targetGrid_visualizer = marker_publisher(self.world)

        self.loop_sleep_time = param['loop_sleep_time']


        
    def plotPath(self):
        x = [self.MDPcore.contiPath[i][0] for i in range(len(self.MDPcore.contiPath))]
        y = [self.MDPcore.contiPath[i][1] for i in range(len(self.MDPcore.contiPath))]
        plt.plot(x, y, marker='o', linestyle='-', color='b')
        plt.title('continuous route')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.show()    

    def run_filter(self):
        
        self.MDPcore.runMDP()
        self.plotPath()

        results = np.zeros((4,7))

        for i in range(5):
            # publish target grid
            self.targetGrid_visualizer.publish_markers()
            # publish target path
            rospy.sleep(self.loop_sleep_time)
        for i in range(len(self.MDPcore.contiPath)):
            self.target_pub.publish_target_path(self.MDPcore.contiPath[i])


        # X, P, particles, particle_weight, mu, Sigma = 0 , 0 , 0 , 0 , 0 , 0
        # for t in range(self.num_step):
        for t in range(10):
            
            # get data for current timestamp
            # motion_command = self.data['motionCommand'][t,:]
            # observation = self.data['observation'][t,:]

            # self.filter_.prediction(motion_command)
            # self.filter_.correction(observation, self.landmarks)
            # self.state_ = self.filter_.getState()
            
            # publisher 



            rospy.sleep(self.loop_sleep_time)

        
        
        return results 
    
    #maybe we can define more specific function to output each step or prediction or correction
        

def main():

    rob_sys = RobotSystem()

    pass

if __name__ == '__main__':
    main()