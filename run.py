from system.RobotSystem import *
from world.world2d import *
from comm.marker_publisher import *

def main():
    print("running...")
    world = world2d()
    robot_system = RobotSystem(world)
    robot_system.run_filter()
    robot_system.plot_res()
    print("done.")

if __name__ == '__main__':
    main()