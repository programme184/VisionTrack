#!/home/yfu/miniconda3/envs/ros/bin/python
import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ur_control.camera_info import CameraInfoReader
from ur_control.transformation import CameraTransformer

class PoseGetter:
    def __init__(self):
        rospy.init_node('pose_getter', anonymous=True)
        self.bridge = CvBridge()

        self.image = None
        self.camera_reader = CameraInfoReader()
        self.camera_info = None
        self.transformer = None

        # Subscribe to live RGB image
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

        # Wait until camera info and pose are available
        rospy.loginfo("Waiting for camera intrinsics and pose...")
        while not rospy.is_shutdown():
            self.camera_info = self.camera_reader.get_camera_info()
            camera_pose = self.camera_reader.get_camera_pose()
            if self.camera_info and camera_pose:
                break
            rospy.sleep(0.1)

        # Initialize transformer (loads from file internally)
        self.transformer = CameraTransformer()

        self.run()

    def image_callback(self, msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr("Image conversion failed: %s", e)

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            depth = self.camera_reader.get_depth_at(x, y)
            if depth and depth > 0:
                world_position = self.transformer.pixel_to_3d_world(x, y, depth)
                print(f"2D pixel: ({x}, {y}) → Depth: {depth:.3f} m → 3D World Pose: {world_position}")
            else:
                print(f"Invalid or missing depth at pixel ({x}, {y})")

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
