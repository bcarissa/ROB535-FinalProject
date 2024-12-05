import yaml

import rospy
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from world.world2d import *
from utils.Landmark import *
from system.RobotSystem import *

class marker_publisher:
    def __init__(self, world):

        # load params
        with open("config/settings.yaml", 'r') as stream:
            param = yaml.safe_load(stream)
        
        marker_topic = param['landmark_topic']        
        self.frame_id = param['marker_frame_id']
        self.world_dim = param['world_dimension']
        self.world = world
        self.grid_size = self.world.grid_size
        self.pub = rospy.Publisher(marker_topic,MarkerArray,queue_size=10)

    def publish_markers(self):

        markerArray = MarkerArray()
        for y in range(self.world.rows):
            for x in range(self.world.cols):
                marker = Marker()
                marker.id = y * self.world.cols + x
                marker.header.frame_id = self.frame_id
                marker.type = marker.CUBE
                marker.action = Marker.DELETE
                marker.action = marker.ADD
                marker.scale.x = self.grid_size
                marker.scale.y = self.grid_size
                marker.scale.z = 0.1

                marker.pose.orientation.w = 1.0
                marker.pose.position.x = x * self.grid_size + self.grid_size / 2
                marker.pose.position.y = y * self.grid_size + self.grid_size / 2
                marker.pose.position.z = 0

                if self.world.fence_grid[y, x] == 1:  # Fence present
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    marker.color.a = 0.8
                elif self.world.fence_grid[y, x] == 2:
                    marker.color.r = 1.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                    path = self.world.path
                    # marker.color.a = (1-(path.index([y,x])/len(path)))*0.3+0.3
                    marker.color.a = 0.3
                else:  # Empty grid
                    marker.color.r = 0.0
                    marker.color.g = 0.0
                    marker.color.b = 1.0
                    marker.color.a = 0.8
                
                markerArray.markers.append(marker)        
        self.pub.publish(markerArray)

        continuousRoute = None
        return continuousRoute


