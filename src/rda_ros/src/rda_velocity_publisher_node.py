#! /usr/bin/env python

import numpy as np
import rospy
import tf
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from math import cos, sin, atan2


class rda_velocity_publisher:
    def __init__(self) -> None:
        # Initialize ROS node
        rospy.init_node("rda_velocity_publisher_node", anonymous=True)
        
        # Publisher for velocity commands
        self.vel_pub = rospy.Publisher("/rda_cmd_vel", Twist, queue_size=10)
        
        # Subscriber for planned trajectory
        rospy.Subscriber("/rda_planned_trajectory", Path, self.planned_traj_callback)
        
        # Parameters
        self.pub_rate = rospy.get_param("~pub_rate", 50.0)  # Hz
        self.interpolation_method = rospy.get_param("~interpolation_method", "linear")
        
        # Initialize tf listener
        self.listener = tf.TransformListener()
        
        # Initialize storage for trajectory
        self.trajectory_poses = []
        self.trajectory_vels = []
        self.trajectory_times = []
        self.last_traj_time = rospy.Time(0)
        self.trajectory_valid = False
        
        # Robot state
        self.robot_state = None
        
        rospy.loginfo("RDA Velocity Publisher initialized")
        
    def planned_traj_callback(self, path_msg):
        """Callback for receiving planned trajectory with velocities"""
        if len(path_msg.poses) < 2:
            rospy.logwarn("Received empty or too short planned trajectory")
            self.trajectory_valid = False
            return
            
        # Extract trajectory data
        self.trajectory_poses = []
        self.trajectory_vels = []
        self.trajectory_times = []
        
        start_time = rospy.Time.now()
        
        for i, pose in enumerate(path_msg.poses):
            # Extract position
            pos = np.array([pose.pose.position.x, pose.pose.position.y])
            
            # Extract orientation (heading)
            quat = (pose.pose.orientation.x, pose.pose.orientation.y, 
                    pose.pose.orientation.z, pose.pose.orientation.w)
            
            # Extract velocities
            linear_vel = pose.pose.position.z
            angular_vel = pose.pose.orientation.x
            
            # Extract time offset
            time_offset = pose.pose.orientation.y
            timestamp = start_time + rospy.Duration(time_offset)
            
            self.trajectory_poses.append(pos)
            self.trajectory_vels.append((linear_vel, angular_vel))
            self.trajectory_times.append(timestamp)
        
        self.last_traj_time = start_time
        self.trajectory_valid = True
        rospy.loginfo(f"Received new trajectory with {len(self.trajectory_poses)} points")
    
    def find_closest_point_on_trajectory(self):
        """Find the closest point on trajectory to current robot position"""
        if not self.trajectory_valid or self.robot_state is None:
            return None, None
            
        robot_pos = self.robot_state[:2]
        
        # Calculate distances to each point
        distances = [np.linalg.norm(robot_pos - pos) for pos in self.trajectory_poses]
        
        # Find closest point
        min_idx = np.argmin(distances)
        
        # If we're at the end of the trajectory, use the last point
        if min_idx >= len(self.trajectory_poses) - 1:
            return min_idx, 1.0
            
        # Check if the next point is a better fit by projecting onto the segment
        p0 = np.array(self.trajectory_poses[min_idx])
        p1 = np.array(self.trajectory_poses[min_idx + 1])
        
        v = p1 - p0  # Vector along the segment
        w = robot_pos - p0  # Vector from p0 to robot
        
        # Project w onto v
        c1 = np.dot(w, v)
        c2 = np.dot(v, v)
        
        # Calculate projection ratio (how far along the segment)
        if c2 == 0:  # Zero length segment
            ratio = 0.0
        else:
            ratio = c1 / c2
            
        if 0.0 <= ratio <= 1.0:
            # Robot is projected onto segment, interpolate
            return min_idx, ratio
        elif ratio < 0.0:
            # Robot is behind segment
            return min_idx, 0.0
        else:
            # Robot is beyond segment
            return min_idx, 1.0
    
    def interpolate_velocity(self, closest_idx, ratio):
        """Interpolate velocity based on closest point and ratio"""
        if closest_idx is None:
            return None
            
        # If we're at the last point or second-to-last with ratio=1.0
        if closest_idx >= len(self.trajectory_vels) - 1 or (closest_idx == len(self.trajectory_vels) - 2 and ratio >= 1.0):
            return self.trajectory_vels[-1]
            
        # Linear interpolation between velocities
        v0_linear, v0_angular = self.trajectory_vels[closest_idx]
        v1_linear, v1_angular = self.trajectory_vels[closest_idx + 1]
        
        linear_vel = v0_linear + ratio * (v1_linear - v0_linear)
        angular_vel = v0_angular + ratio * (v1_angular - v0_angular)
        
        return (linear_vel, angular_vel)
    
    def check_trajectory_expired(self):
        """Check if the current trajectory has expired and needs to be reset"""
        if not self.trajectory_valid:
            return True
            
        # If trajectory is too old, consider it expired
        current_time = rospy.Time.now()
        if (current_time - self.last_traj_time).to_sec() > 1.0:  # 1 second expiration
            rospy.logwarn_throttle(2, "Trajectory has expired - waiting for new one")
            self.trajectory_valid = False
            return True
            
        return False
    
    def convert_to_twist(self, velocities):
        """Convert velocity tuple to Twist message"""
        if velocities is None:
            return None
            
        linear_vel, angular_vel = velocities
        
        vel = Twist()
        vel.linear.x = linear_vel
        vel.angular.z = angular_vel
        
        return vel
    
    def publish_velocities(self):
        """Main loop to publish velocities at a high rate"""
        rate = rospy.Rate(self.pub_rate)
        
        rospy.loginfo(f"Starting velocity publisher at {self.pub_rate} Hz")
        
        zero_vel = Twist()
        
        while not rospy.is_shutdown():
            # Check if trajectory is valid
            if self.check_trajectory_expired():
                self.vel_pub.publish(zero_vel)
                rate.sleep()
                continue

            # Find where we are on the trajectory
            closest_idx, ratio = self.find_closest_point_on_trajectory()
            
            # If we couldn't find a good point, stop
            if closest_idx is None:
                self.vel_pub.publish(zero_vel)
                rate.sleep()
                continue
                
            # Interpolate velocity
            velocities = self.interpolate_velocity(closest_idx, ratio)
            
            # Convert to Twist and publish
            vel_msg = self.convert_to_twist(velocities)
            self.vel_pub.publish(vel_msg)
            
            rate.sleep()
    
    @staticmethod
    def quat_to_yaw_list(quater):
        """Convert quaternion to yaw angle"""
        x = quater[0]
        y = quater[1]
        z = quater[2]
        w = quater[3]

        raw = atan2(2 * (w * z + x * y), 1 - 2 * (pow(z, 2) + pow(y, 2)))

        return raw
    
    @staticmethod
    def quat_to_yaw(quater):
        """Convert quaternion to yaw angle"""
        x = quater.x
        y = quater.y
        z = quater.z
        w = quater.w

        raw = atan2(2 * (w * z + x * y), 1 - 2 * (pow(z, 2) + pow(y, 2)))

        return raw


if __name__ == "__main__":
    try:
        vel_publisher = rda_velocity_publisher()
        vel_publisher.publish_velocities()
    except rospy.ROSInterruptException:
        pass