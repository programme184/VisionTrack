#!/home/yfu/miniconda3/envs/ros/bin/python
import os
import json
import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ur_control.transformation import CameraTransformer

class PoseGetter:
    def __init__(self):
        if not rospy.core.is_initialized():
            rospy.init_node('pose_getter', anonymous=True)
        self.bridge = CvBridge()

        self.image = None
        self.camera_info = None  # dict with fx, fy, cx, cy
        self.camera_pose = None  # dict with position and orientation
        self.transformer = None

        # Subscribe to live RGB image
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

        # Load camera intrinsics and pose from config
        rospy.loginfo("Loading camera intrinsics and pose from config...")
        self.load_camera_config()

        # Initialize transformer with loaded config
        self.transformer = CameraTransformer(self.camera_info, self.camera_pose)

        self.run()

    def load_camera_config(self):
        config_path = os.path.join(
            os.path.dirname(__file__), '../camera_config.json'
        )
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.camera_info = config.get('intrinsics')
        self.camera_pose = config.get('pose')

        if not self.camera_info or not self.camera_pose:
            rospy.logerr("Missing 'intrinsics' or 'pose' in camera_config.json")
            rospy.signal_shutdown("Camera config incomplete")

    def image_callback(self, msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr("Image conversion failed: %s", e)

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            depth = self.get_depth_at(x, y)
            if depth and depth > 0:
                world_position = self.transformer.pixel_to_3d_world(x, y, depth)
                print(f"2D pixel: ({x}, {y}) → Depth: {depth:.3f} m → 3D World Pose: {world_position}")
            else:
                print(f"Invalid or missing depth at pixel ({x}, {y})")

    def get_depth_at(self, x, y):
        # Placeholder stub; update this to get actual depth from aligned depth image
        return 0.8  # example fixed depth in meters

    def run(self):
        cv2.namedWindow("Click to get 3D Pose")
        cv2.setMouseCallback("Click to get 3D Pose", self.on_mouse_click)

        while not rospy.is_shutdown():
            if self.image is not None:
                cv2.imshow("Click to get 3D Pose", self.image)
                if cv2.waitKey(10) == 27:  # ESC to quit
                    break
        cv2.destroyAllWindows()

if __name__ == '__main__':
    PoseGetter()
