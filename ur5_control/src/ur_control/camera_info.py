#!/home/yfu/miniconda3/envs/ros/bin/python
import rospy
from sensor_msgs.msg import CameraInfo
from gazebo_msgs.msg import LinkStates
import numpy as np
import json
import os

class CameraInfoReader:
    def __init__(self, camera_link_name='kinect_0::link'):
        rospy.init_node('camera_info_reader', anonymous=True)

        self.camera_info = None
        self.camera_pose = None
        self.camera_link_name = camera_link_name

        self.info_received = False
        self.pose_received = False

        # Subscribers
        rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.info_callback)
        rospy.Subscriber('/gazebo/link_states', LinkStates, self.link_states_callback)

    def info_callback(self, msg):
        if not self.info_received:
            self.camera_info = {
                'fx': msg.K[0],
                'fy': msg.K[4],
                'cx': msg.K[2],
                'cy': msg.K[5]
            }
            self.info_received = True
            rospy.loginfo("Camera intrinsics received.")

    def link_states_callback(self, msg):
        if not self.pose_received:
            try:
                index = msg.name.index(self.camera_link_name)
                pose = msg.pose[index]
                position = pose.position
                orientation = pose.orientation
                self.camera_pose = {
                    'position': [position.x, position.y, position.z],
                    'orientation': [orientation.x, orientation.y, orientation.z, orientation.w]
                }
                self.pose_received = True
                rospy.loginfo("Camera world pose received.")
            except ValueError:
                pass  # camera link not found

    def is_ready(self):
        return self.info_received and self.pose_received

    def save_to_file(self, path='./src/ur5_gripper_sim/ur5_control/camera_config.json'):
        data = {
            'intrinsics': self.camera_info,
            'pose': self.camera_pose
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        rospy.loginfo(f"Camera config saved to {os.path.abspath(path)}")

if __name__ == "__main__":
    reader = CameraInfoReader()
    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        if reader.is_ready():
            reader.save_to_file()
            break
        rate.sleep()
