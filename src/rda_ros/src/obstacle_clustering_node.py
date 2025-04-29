#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
from geometry_msgs.msg import Point32, TwistWithCovariance, Twist
import tf
from collections import namedtuple
from sklearn.cluster import DBSCAN
import cv2
from math import cos, sin

rda_obs_tuple = namedtuple(
    "rda_obs_tuple", "center radius vertex cone_type velocity"
)

class ObstacleClusteringNode:
    def __init__(self):
        rospy.init_node("obstacle_clustering_node", anonymous=True)

        # ROS parameters
        self.scan_eps = rospy.get_param("~scan_eps", 0.2)
        self.scan_min_samples = rospy.get_param("~scan_min_samples", 6)
        self.target_frame = rospy.get_param("~target_frame", "map")
        self.lidar_frame = rospy.get_param("~lidar_frame", "lidar_link")

        # Publisher
        self.obs_pub = rospy.Publisher("/rda_obstacles", ObstacleArrayMsg, queue_size=10)

        # Subscriber
        rospy.Subscriber("/scan", LaserScan, self.scan_callback)

        # TF listener
        self.listener = tf.TransformListener()

    def scan_callback(self, scan_data):
        ranges = np.array(scan_data.ranges)
        angles = np.linspace(scan_data.angle_min, scan_data.angle_max, len(ranges))

        point_list = []
        for i in range(len(ranges)):
            scan_range = ranges[i]
            angle = angles[i]
            if scan_range < (scan_data.range_max - 0.01):
                point = np.array(
                    [[scan_range * cos(angle)], [scan_range * sin(angle)]]
                )
                point_list.append(point)

        if len(point_list) < 3:
            rospy.loginfo_throttle(1, "No obstacles are converted to polygon")
            self.publish_obstacles([])
            return

        # Get transform from lidar to target frame
        try:
            trans, rot = self.listener.lookupTransform(
                self.target_frame, self.lidar_frame, rospy.Time(0)
            )
            yaw = self.quat_to_yaw_list(rot)
            x, y = trans[0], trans[1]
            l_trans, l_R = self.get_transform(np.array([x, y, yaw]).reshape(3, 1))
        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ):
            rospy.loginfo_throttle(
                1,
                "Waiting for tf transform from {} to {}".format(
                    self.lidar_frame, self.target_frame
                ),
            )
            return

        # Perform DBSCAN clustering
        point_array = np.hstack(point_list).T
        labels = DBSCAN(
            eps=self.scan_eps, min_samples=self.scan_min_samples
        ).fit_predict(point_array)

        obstacle_list = []
        for label in np.unique(labels):
            if label == -1:
                continue
            point_array2 = point_array[labels == label]
            rect = cv2.minAreaRect(point_array2.astype(np.float32))
            box = cv2.boxPoints(rect)
            vertices = box.T
            global_vertices = l_R @ vertices + l_trans
            obstacle_list.append(
                rda_obs_tuple(None, None, global_vertices, "Rpositive", 0)
            )

        self.publish_obstacles(obstacle_list)

    def publish_obstacles(self, obstacle_list):
        obs_array = ObstacleArrayMsg()
        obs_array.header.stamp = rospy.get_rostime()
        obs_array.header.frame_id = self.target_frame

        for obs in obstacle_list:
            if obs.vertex is not None:  # Polygon obstacle
                obs_msg = ObstacleMsg()
                obs_msg.header.stamp = rospy.get_rostime()
                obs_msg.header.frame_id = self.target_frame
                for i in range(obs.vertex.shape[1]):
                    point = Point32()
                    point.x = obs.vertex[0, i]
                    point.y = obs.vertex[1, i]
                    point.z = 0.0
                    obs_msg.polygon.points.append(point)
                # Use TwistWithCovariance for velocities
                obs_msg.velocities = TwistWithCovariance()
                obs_msg.velocities.twist = Twist()
                # Set zero velocity (as per rda_obs_tuple)
                obs_msg.velocities.twist.linear.x = 0.0
                obs_msg.velocities.twist.linear.y = 0.0
                obs_msg.velocities.twist.angular.z = 0.0
                # Set empty covariance (36-element array of zeros)
                obs_msg.velocities.covariance = [0.0] * 36
                obs_array.obstacles.append(obs_msg)

        self.obs_pub.publish(obs_array)

    def get_transform(self, state):
        if state.shape == (2, 1):
            rot = np.array([[1, 0], [0, 1]])
            trans = state[0:2]
        else:
            rot = np.array(
                [
                    [cos(state[2, 0]), -sin(state[2, 0])],
                    [sin(state[2, 0]), cos(state[2, 0])],
                ]
            )
            trans = state[0:2]
        return trans, rot

    @staticmethod
    def quat_to_yaw_list(quater):
        x, y, z, w = quater
        raw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (z**2 + y**2))
        return raw

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        node = ObstacleClusteringNode()
        node.run()
    except rospy.ROSInterruptException:
        pass