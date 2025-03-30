#!/usr/bin/env python3
import rospy
import os
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ImageCaptureSaver:
    def __init__(self, save_dir="../../ImageData", max_images=100):
        self.bridge = CvBridge()
        self.save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), save_dir))
        self.rgb_dir = os.path.join(self.save_dir, "rgb")
        self.depth_dir = os.path.join(self.save_dir, "depth")

        self.max_images = max_images
        self.rgb_count = 0
        self.depth_count = 0

        self._ensure_directories()

        rospy.init_node("image_saver", anonymous=True)

        self.rgb_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.save_rgb, queue_size=1)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.save_depth, queue_size=1)

        rospy.loginfo(f"Saving up to {self.max_images} RGB and Depth images to {self.save_dir}")

    def _ensure_directories(self):
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)

    def save_rgb(self, msg):
        if self.rgb_count >= self.max_images:
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            filename = os.path.join(self.rgb_dir, f"rgb_{rospy.Time.now().to_nsec()}.png")
            cv2.imwrite(filename, cv_image)
            self.rgb_count += 1
            self._check_shutdown()
        except Exception as e:
            rospy.logerr(f"Failed to save RGB image: {e}")

    def save_depth(self, msg):
        if self.depth_count >= self.max_images:
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            filename = os.path.join(self.depth_dir, f"depth_{rospy.Time.now().to_nsec()}.png")
            norm_image = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX)
            norm_image = norm_image.astype('uint8')
            cv2.imwrite(filename, norm_image)
            self.depth_count += 1
            self._check_shutdown()
        except Exception as e:
            rospy.logerr(f"Failed to save depth image: {e}")

    def _check_shutdown(self):
        if self.rgb_count >= self.max_images and self.depth_count >= self.max_images:
            rospy.loginfo("All requested images saved. Shutting down.")
            rospy.signal_shutdown("Done saving images.")

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    saver = ImageCaptureSaver(save_dir="../../ImageData", max_images=50)
    saver.spin()
