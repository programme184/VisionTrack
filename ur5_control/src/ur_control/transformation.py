import numpy as np
import json
import os
from scipy.spatial.transform import Rotation as R

class CameraTransformer:
    def __init__(self, intrinsics=None, pose=None, config_path=None):
        # Load from file only if no intrinsics/pose provided
        if intrinsics is None or pose is None:
            if config_path is None:
                config_path = os.path.join(
                    os.path.dirname(__file__),
                    '../camera_config.json'
                )
            config_path = os.path.abspath(config_path)
            with open(config_path, 'r') as f:
                config = json.load(f)
            intrinsics = config['intrinsics']
            pose = config['pose']

        # Intrinsics
        self.fx = intrinsics['fx']
        self.fy = intrinsics['fy']
        self.cx = intrinsics['cx']
        self.cy = intrinsics['cy']

        # Camera pose in world frame
        self.camera_position = np.array(pose['position'])
        self.camera_orientation = R.from_quat(pose['orientation'])  # [x, y, z, w]

    def pixel_to_3d_camera(self, x, y, depth):
        """Convert pixel to 3D point in the camera coordinate frame."""
        X = (x - self.cx) * depth / self.fx
        Y = (y - self.cy) * depth / self.fy
        Z = depth
        return np.array([X, Y, Z])

    def pixel_to_3d_world(self, x, y, depth):
        """Convert pixel to 3D point in the world coordinate frame."""
        point_camera = self.pixel_to_3d_camera(x, y, depth)
        point_world = self.camera_orientation.apply(point_camera) + self.camera_position
        return point_world

if __name__ == "__main__":
    # Test standalone usage
    transformer = CameraTransformer()
    world_point = transformer.pixel_to_3d_world(320, 240, 1.0)
    print("3D World Coordinate:", world_point)
