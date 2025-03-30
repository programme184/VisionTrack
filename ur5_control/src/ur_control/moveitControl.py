#!/home/yfu/miniconda3/envs/ros/bin/python

import sys
import rospy
import moveit_commander
import geometry_msgs.msg
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState

class BasicRobotController:
    def __init__(self):
        """
        Initialize the ROS node, the MoveIt interfaces, and store
        references to the arm and gripper move groups.
        """
        # Initialize moveit_commander and rospy node
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('basic_robot_control', anonymous=True)

        # Create the RobotCommander (interface to the robot as a whole)
        self.robot = moveit_commander.RobotCommander()

        # Create the PlanningSceneInterface (interface to the world around the robot)
        self.scene = moveit_commander.PlanningSceneInterface()
        self.arm_group = moveit_commander.MoveGroupCommander("manipulator")
        self.gripper_group = moveit_commander.MoveGroupCommander("gripper")

        # (Optional) Set planning parameters
        self.arm_group.set_planning_time(5.0)
        self.arm_group.set_num_planning_attempts(10)

        rospy.loginfo("BasicRobotController initialized.")

    def get_current_robot_state(self):
        """
        Returns the current state of the entire robot (joints, etc.).
        """
        return self.robot.get_current_state()

    def get_current_arm_joints(self):
        """
        Returns the current joint values of the arm move group.
        """
        return self.arm_group.get_current_joint_values()

    def get_current_pose(self):
        """
        Returns the current end-effector pose of the arm move group.
        """
        return self.arm_group.get_current_pose().pose

    def set_arm_joint_values(self, joint_positions):
        """
        Moves the arm to the specified joint positions.
        :param joint_positions: list of joint angles (in radians) for each joint in 'manipulator'.
        """
        # Set the joint target
        self.arm_group.set_joint_value_target(joint_positions)

        # Plan and execute
        plan = self.arm_group.go(wait=True)
        self.arm_group.stop()  # Ensure no residual movement
        rospy.loginfo("Arm moved to joint positions: %s", joint_positions)
        self.arm_group.clear_pose_targets()
        return plan

    def set_arm_pose(self, pose):
        """
        Moves the arm's end-effector to the specified pose using IK and planning.
        :param pose: geometry_msgs/Pose specifying desired end-effector pose.
        """
        # Set the pose target
        self.arm_group.set_pose_target(pose)

        # Plan and execute
        plan = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
        rospy.loginfo("Arm moved to pose: %s", pose)
        return plan

    def get_gripper_state(self):
        """
        Returns the current joint values of the gripper move group.
        """
        return self.gripper_group.get_current_joint_values()

    def set_gripper(self, joint_positions):
        """
        Moves the gripper joints to the specified positions.
        :param joint_positions: list of joint angles/positions for the gripper joints.
                               For a simple parallel gripper, you might have 1 or 2 joints.
        """
        self.gripper_group.set_joint_value_target(joint_positions)
        plan = self.gripper_group.go(wait=True)
        self.gripper_group.stop()
        rospy.loginfo("Gripper moved to joint positions: %s", joint_positions)
        return plan

    def example_inverse_kinematics(self, target_pose):
        """
        Example of how to manually compute an IK solution. 
        Typically, you'd just use set_pose_target() which does IK internally.
        But if you need to explicitly get the IK solution, you can use
        get_current_state() and set the joint state in the RobotState,
        then call set_from_ik().
        """
        # 1) Create a RobotState with the current state
        current_state = self.robot.get_current_state()
        
        # 2) Use the MoveGroupCommander to get an IK solution
        #    set_from_ik returns a list of joint values if successful, otherwise None
        joint_solutions = self.arm_group.set_from_ik(current_state,
                                                     target_pose,
                                                     end_effector_link=self.arm_group.get_end_effector_link(),
                                                     attempts=10,
                                                     timeout=0.5)
        if joint_solutions:
            rospy.loginfo("Found IK solution: %s", joint_solutions)
            # You could then set these joint values directly
            self.arm_group.set_joint_value_target(joint_solutions)
            plan = self.arm_group.go(wait=True)
            self.arm_group.stop()
            self.arm_group.clear_pose_targets()
            return plan
        else:
            rospy.logwarn("No IK solution found for the given pose.")
            return None

    def shutdown(self):
        """
        Shutdown MoveIt cleanly.
        """
        moveit_commander.roscpp_shutdown()
        rospy.loginfo("Shutting down BasicRobotController.")

def main():
    controller = BasicRobotController()

    # 1) Get and print the current robot state
    current_state = controller.get_current_robot_state()
    rospy.loginfo("Current robot state:\n%s", current_state)

    # 2) Print current arm joint positions
    current_arm_joints = controller.get_current_arm_joints()
    rospy.loginfo("Current arm joint positions: %s", current_arm_joints)

    # 3) Print current end-effector pose
    current_pose = controller.get_current_pose()
    rospy.loginfo("Current end-effector pose: %s", current_pose)

    # 4) Move the arm by joint angles (example values!)
    #    Adjust these to something valid for your robot
    joint_goal = [1.68, -1.57, 1.267, -1.57, 0.9, 0.0]
    controller.set_arm_joint_values(joint_goal)

    # 5) Move the arm to a pose target
    # pose_goal = geometry_msgs.msg.Pose()
    # pose_goal.orientation.w = 1.0
    # pose_goal.position.x = 0.4
    # pose_goal.position.y = 0.0
    # pose_goal.position.z = 0.3
    # controller.set_arm_pose(pose_goal)

    # 6) Get the gripper state
    current_gripper = controller.get_gripper_state()
    rospy.loginfo("Current gripper state: %s", current_gripper)

    # 7) Set the gripper (example: open vs close)
    #    For a simple parallel gripper with a single joint, you'd do something like:
    #    controller.set_gripper([0.0])    # Open
    #    controller.set_gripper([0.8])    # Close
    #    Adjust for however many joints your gripper has.
    # controller.set_gripper([0.0])  # Example "open" position
    # rospy.sleep(1.0)
    # controller.set_gripper([0.8])  # Example "close" position

    # # 8) Example of explicitly calling an IK function
    # #    (Normally you'd just do set_arm_pose() above)
    # new_pose_goal = geometry_msgs.msg.Pose()
    # new_pose_goal.orientation.w = 1.0
    # new_pose_goal.position.x = 0.5
    # new_pose_goal.position.y = 0.2
    # new_pose_goal.position.z = 0.2
    # controller.example_inverse_kinematics(new_pose_goal)

    # Shutdown
    controller.shutdown()

if __name__ == '__main__':
    main()
