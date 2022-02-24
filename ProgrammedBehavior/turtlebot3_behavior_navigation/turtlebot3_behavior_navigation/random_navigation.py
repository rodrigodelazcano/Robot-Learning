import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from random import choices, random

class RandomBehavior(Node):
    def __init__(self):
        super().__init__(node_name='random_behavior_node')
        self.vel_publisher_ = self.create_publisher(Twist, 'cmd_vel', 1)

        qos_policy = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                                          history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                                          depth=1)

        self.lidar_subscriber_ = self.create_subscription(LaserScan, 'scan', self.lidar_callback, qos_profile=qos_policy)
        self.lidar_subscriber_ = self.create_subscription(LaserScan, 'odom', self.odom_callback, qos_profile=qos_policy)

        self.turtlebot_radius = 220 # mm
        self.angle_roi = 90 # region of interest to scan for obstacles in degrees
        self.angle_roi_radians = math.radians(90)
        self.clearance = 200 # mm

        self.save_distance = self.turtlebot_radius / math.cos(math.radians((180 - self.angle_roi) / 2)) + self.clearance # if any laser ray from the roi has lower value perform turning

        self.forward_vel = Twist()
        self.forward_vel.linear.x = 0.5 # m/s

        self.left_turn_vel = Twist()
        self.left_turn_vel.angular.z = 0.5

        self.right_turn_vel = Twist()
        self.right_turn_vel.angular.z = - 0.5

        self.stop_vel = Twist()

        # random exploration actions
        self.random_vels = [self.forward_vel, self.left_turn_vel, self.right_turn_vel]
        # random exploration probabilities of each action
        self.probabilities = [0.6, 0.3, 0.1]

    def lidar_callback(self, msg):
        self.lidar_data = msg

        max_angle = msg.angle_max
        min_angle = msg.angle_min
        resolution = msg.angle_increment

        max_rays = round(max_angle/ resolution)

        # get front values of lidar
        lidar_idx = int(round(self.angle_roi_radians/2/resolution))

        left_idx = np.linspace(0,lidar_idx,lidar_idx+1, dtype=int)
        right_idx = np.linspace(max_rays - lidar_idx, max_rays, lidar_idx+1, dtype=int)

        left_lidar_data = list(self.lidar_data.ranges[left_idx[0]:left_idx[-1]])
        right_lidar_data = list(self.lidar_data.ranges[right_idx[0]:right_idx[-1]])

        if any(x < self.save_distance/1000 for x in left_lidar_data):
            obstacle_left = True
        else:
            obstacle_left = False

        if any(x < self.save_distance/1000 for x in right_lidar_data):
            obstacle_right = True
        else:
            obstacle_right = False

        # if any of them passes certain value turn
        # direction of turning depends on the laser ray being active

        # if no obstacles detected in the field of view => randomly explore
        if (obstacle_right or obstacle_left) == False:
            random_vel = choices(self.random_vels, self.probabilities)[0]
            self.vel_publisher_.publish(random_vel)
        #if obstacle on left field of view => turn right
        elif obstacle_left == True and obstacle_right == False:
            self.vel_publisher_.publish(self.right_turn_vel)
        # if obstacle on right field of view => turn left
        elif obstacle_left == False and obstacle_right == True:
            self.vel_publisher_.publish(self.left_turn_vel)
        # if obstacle on both sides of field of view => turn right
        elif (obstacle_left and obstacle_right) == True:
            self.vel_publisher_.publish(self.right_turn_vel)
        else:
            self.vel_publisher_.publish(self.stop_vel)

def main(args=None):
    rclpy.init(args=args)

    random_behavior_node = RandomBehavior()

    rclpy.spin(random_behavior_node)

    random_behavior_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()