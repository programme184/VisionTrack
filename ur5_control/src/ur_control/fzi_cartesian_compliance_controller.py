# The MIT License (MIT)
#
# Copyright (c) 2023 Cristian Beltran
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Author: Cristian Beltran

import collections
import threading
import types
import rospy
import numpy as np

from ur5_control.arm import Arm
from ur_control import conversions
from ur5_control.constants import JOINT_TRAJECTORY_CONTROLLER, CARTESIAN_COMPLIANCE_CONTROLLER, ExecutionResult

from geometry_msgs.msg import WrenchStamped, PoseStamped

import dynamic_reconfigure.client


def is_more_extreme(value, target):
    if (np.all(value > 0) and np.all(target > 0)):
        return np.all(value > target)
    elif (np.all(value < 0) and np.all(target < 0)):
        return np.all(value < target)
    return False


def convert_selection_matrix_to_parameters(selection_matrix):
    return {
        "stiffness":
        {
            "sel_x": selection_matrix[0],
            "sel_y": selection_matrix[1],
            "sel_z": selection_matrix[2],
            "sel_ax": selection_matrix[3],
            "sel_ay": selection_matrix[4],
            "sel_az": selection_matrix[5],
        }
    }


def convert_stiffness_to_parameters(stiffness):
    return {
        "stiffness":
        {
            "trans_x": stiffness[0],
            "trans_y": stiffness[1],
            "trans_z": stiffness[2],
            "rot_x": stiffness[3],
            "rot_y": stiffness[4],
            "rot_z": stiffness[5],
        }
    }


def convert_pd_gains_to_parameters(p_gains, d_gains=[0, 0, 0, 0, 0, 0]):
    return {
        "trans_x": {"p": p_gains[0], "d": d_gains[0]},
        "trans_y": {"p": p_gains[1], "d": d_gains[1]},
        "trans_z": {"p": p_gains[2], "d": d_gains[2]},
        "rot_x": {"p": p_gains[3], "d": d_gains[3]},
        "rot_y": {"p": p_gains[4], "d": d_gains[4]},
        "rot_z": {"p": p_gains[5], "d": d_gains[5]}
    }


def switch_cartesian_controllers(func):
    '''Decorator that switches from cartesian to joint trajectory controllers and back'''

    def wrap(*args, **kwargs):
        if not args[0].auto_switch_controllers:
            return func(*args, **kwargs)

        args[0].activate_cartesian_controller()

        try:
            res = func(*args, **kwargs)
        except Exception as e:
            rospy.logerr("Exception: %s" % e)
            res = ExecutionResult.DONE

        args[0].activate_joint_trajectory_controller()

        return res
    return wrap


class CompliantController(Arm):
    def __init__(self,
                 **kwargs):
        """ Compliant controller using FZI Cartesian Compliance controllers """
        Arm.__init__(self, **kwargs)

        self.is_gazebo_sim = False
        if rospy.has_param("use_gazebo_sim"):
            self.is_gazebo_sim = True

        self.rate = rospy.Rate(self.joint_traj_controller.rate)
        self.min_dt = 1. / self.joint_traj_controller.rate

        self.auto_switch_controllers = True  # Safety switching back to safe controllers

        self.current_target_pose = np.zeros(7)
        self.current_wrench_pose = np.zeros(6)

        # Monitor external goals
        rospy.Subscriber('%s%s/target_frame' % (self.ns, CARTESIAN_COMPLIANCE_CONTROLLER), PoseStamped, self.target_pose_cb)
        rospy.Subscriber('%s%s/target_wrench' % (self.ns, CARTESIAN_COMPLIANCE_CONTROLLER), WrenchStamped, self.target_wrench_cb)

        self.cartesian_target_pose_pub = rospy.Publisher('%s%s/target_frame' % (self.ns, CARTESIAN_COMPLIANCE_CONTROLLER), PoseStamped, queue_size=10.0)
        self.cartesian_target_wrench_pub = rospy.Publisher('%s%s/target_wrench' % (self.ns, CARTESIAN_COMPLIANCE_CONTROLLER), WrenchStamped, queue_size=10.0)

        self.dyn_config_clients = {
            "trans_x": dynamic_reconfigure.client.Client("%s%s/pd_gains/trans_x" % (self.ns, CARTESIAN_COMPLIANCE_CONTROLLER), timeout=10),
            "trans_y": dynamic_reconfigure.client.Client("%s%s/pd_gains/trans_y" % (self.ns, CARTESIAN_COMPLIANCE_CONTROLLER), timeout=10),
            "trans_z": dynamic_reconfigure.client.Client("%s%s/pd_gains/trans_z" % (self.ns, CARTESIAN_COMPLIANCE_CONTROLLER), timeout=10),
            "rot_x": dynamic_reconfigure.client.Client("%s%s/pd_gains/rot_x" % (self.ns, CARTESIAN_COMPLIANCE_CONTROLLER), timeout=10),
            "rot_y": dynamic_reconfigure.client.Client("%s%s/pd_gains/rot_y" % (self.ns, CARTESIAN_COMPLIANCE_CONTROLLER), timeout=10),
            "rot_z": dynamic_reconfigure.client.Client("%s%s/pd_gains/rot_z" % (self.ns, CARTESIAN_COMPLIANCE_CONTROLLER), timeout=10),

            "stiffness": dynamic_reconfigure.client.Client("%s%s/stiffness" % (self.ns, CARTESIAN_COMPLIANCE_CONTROLLER), timeout=10),

            "hand_frame_control": dynamic_reconfigure.client.Client("%s%s/force" % (self.ns, CARTESIAN_COMPLIANCE_CONTROLLER), timeout=10),

            "solver": dynamic_reconfigure.client.Client("%s%s/solver" % (self.ns, CARTESIAN_COMPLIANCE_CONTROLLER), timeout=10),

            "end_effector_link": dynamic_reconfigure.client.Client("%s%s" % (self.ns, CARTESIAN_COMPLIANCE_CONTROLLER), timeout=10),
        }
        self.param_update_queue = collections.deque(maxlen=15)
        self.update_thread = None
        self.update_lock = threading.Lock()
        self.update_condition = threading.Condition()
        self.update_thread_stopped = False
        self.async_mode = False

        self.set_hand_frame_control(False)
        self.set_end_effector_link(self.ee_link)
        self.min_scale_error = 1.5

        rospy.on_shutdown(self.activate_joint_trajectory_controller)

    def __del__(self):
        # wake up thread and stop it
        if hasattr(self, 'update_condition'):
            with self.update_condition:
                self.update_thread_stopped = True
                self.update_condition.notify_all()

    def target_pose_cb(self, data):
        self.current_target_pose = conversions.from_pose_to_list(data.pose)

    def target_wrench_cb(self, data):
        self.current_target_wrench = conversions.from_wrench(data.wrench)

    def activate_cartesian_controller(self):
        return self.controller_manager.switch_controllers(controllers_on=[CARTESIAN_COMPLIANCE_CONTROLLER],
                                                          controllers_off=[JOINT_TRAJECTORY_CONTROLLER])

    def activate_joint_trajectory_controller(self):
        return self.controller_manager.switch_controllers(controllers_on=[JOINT_TRAJECTORY_CONTROLLER],
                                                          controllers_off=[CARTESIAN_COMPLIANCE_CONTROLLER])

    def set_cartesian_target_wrench(self, wrench: list):
        # Publish the target wrench
        try:
            target_wrench = WrenchStamped()
            target_wrench.header.frame_id = self.base_link
            target_wrench.wrench = conversions.to_wrench(wrench)
            self.cartesian_target_wrench_pub.publish(target_wrench)
        except Exception as e:
            rospy.logerr("Fail to set_target_wrench(): %s" % e)

    def set_cartesian_target_pose(self, pose: list):
        # Publish the target pose
        try:
            target_pose = conversions.to_pose_stamped(self.base_link, pose)
            self.cartesian_target_pose_pub.publish(target_pose)
        except Exception as e:
            rospy.logerr("Fail to set_target_pose(): %s" % e)

    def publish_parameter_update(self, parameters):
        try:
            for param in parameters.keys():
                self.dyn_config_clients[param].update_configuration(parameters[param])
        except Exception as e:
            rospy.logerr_throttle(1, f"failed publish_parameter_update {e}")
            pass

    def __update_controller_parameter_loop__(self):
        while not rospy.is_shutdown():
            if self.update_thread_stopped:
                return

            # Sleep until new request is available
            if not self.param_update_queue:
                with self.update_condition:
                    self.update_condition.wait(timeout=0.5)
                return
            else:
                # Lock queue update
                with self.update_lock:
                    parameters = self.param_update_queue.pop()
                if parameters:
                    self.publish_parameter_update(parameters)
                    parameters = None

    def update_controller_parameters(self, parameters: dict):
        if self.async_mode:
            with self.update_lock:
                self.param_update_queue.append(parameters)
            if self.update_thread is None or not self.update_thread.is_alive():
                del self.update_thread
                self.update_thread = threading.Thread(target=self.__update_controller_parameter_loop__)
                self.update_thread.start()
            with self.update_condition:
                self.update_condition.notify()
        else:
            self.publish_parameter_update(parameters)

    def update_selection_matrix(self, selection_matrix):
        parameters = convert_selection_matrix_to_parameters(selection_matrix)
        self.update_controller_parameters(parameters)

    def update_pd_gains(self, p_gains, d_gains=[0, 0, 0, 0, 0, 0]):
        parameters = convert_pd_gains_to_parameters(p_gains, d_gains)
        self.update_controller_parameters(parameters)

    def update_stiffness(self, stiffness):
        parameters = convert_stiffness_to_parameters(stiffness)
        self.update_controller_parameters(parameters)

    def set_control_mode(self, mode="parallel"):
        parameters = {"stiffness": {}}
        if mode == "parallel":
            parameters["stiffness"].update({"use_parallel_force_position_control": True})
        elif mode == "spring-mass-damper":
            parameters["stiffness"].update({"use_parallel_force_position_control": False})
        else:
            raise ValueError("Unknown control mode %s" % mode)
        self.update_controller_parameters(parameters)

    def set_position_control_mode(self, enable=True):
        parameters = convert_selection_matrix_to_parameters(np.ones(6))
        parameters["stiffness"].update({"use_parallel_force_position_control": enable})
        self.update_controller_parameters(parameters)

    def set_hand_frame_control(self, enable):
        parameters = {"hand_frame_control": {"hand_frame_control": enable}}
        self.update_controller_parameters(parameters)

    def set_end_effector_link(self, end_effector_link):
        """ Change the end_effector_link used in the Cartesian Compliance Controllers"""
        parameters = {"end_effector_link": {"end_effector_link": end_effector_link}}
        self.update_controller_parameters(parameters)

    def set_solver_parameters(self, error_scale=None, iterations=None, publish_state_feedback=None):
        parameters = {"solver": {}}
        if error_scale:
            error_scale = error_scale if not self.is_gazebo_sim else error_scale * 0.01
            parameters["solver"].update({"error_scale": round(error_scale, 4)})
        if iterations:
            parameters["solver"].update({"iterations": iterations})
        if publish_state_feedback:
            parameters["solver"].update({"publish_state_feedback": publish_state_feedback})
        self.update_controller_parameters(parameters)

    def wait_for_robot_to_stop(self, wait_time=5):
        remaining_time = wait_time
        start_time = rospy.get_time()

        prev_state = self.joint_angles()

        no_motion_count = 0

        rate = rospy.Rate(500)

        while remaining_time > 0 and no_motion_count < 3:
            rate.sleep()
            remaining_time = wait_time - (rospy.get_time() - start_time)
            curr_state = self.joint_angles()
            if np.allclose(prev_state, curr_state, atol=0.0001):
                no_motion_count += 1
            else:
                no_motion_count = 0

    @switch_cartesian_controllers
    def execute_compliance_control(self, trajectory: np.array, target_wrench: np.array, max_force_torque: list,
                                   duration: float, stop_on_target_force=False, termination_criteria=None,
                                   auto_stop=True, func=None, scale_up_error=False, max_scale_error=None,
                                   relative_to_ee=False, stop_at_wrench=None):

        # Space out the trajectory points
        trajectory = trajectory.reshape((-1, 7))  # Assuming this format [x,y,z,qx,qy,qz,qw]
        step_duration = max(self.min_dt, duration / float(trajectory.shape[0]))
        trajectory_index = 0

        # loop throw target trajectory
        initial_time = rospy.get_time()
        step_initial_time = rospy.get_time()

        result = ExecutionResult.DONE
        if stop_on_target_force and stop_at_wrench is None:
            raise ValueError("'stop_at_wrench' not specify when requesting 'stop_on_target_force'")

        if stop_on_target_force:
            stop_at_wrench = np.array(stop_at_wrench)
            stop_target_wrench_mask = np.flatnonzero(stop_at_wrench)
            rospy.loginfo_throttle(1, 'TARGET F/T {}'.format(np.round(stop_at_wrench[stop_target_wrench_mask], 2)))

        # Publish target wrench only once
        self.set_cartesian_target_wrench(target_wrench)

        # Publish first trajectory point
        self.set_cartesian_target_pose(trajectory[trajectory_index])

        if scale_up_error and max_scale_error:
            self.sliding_error(trajectory[trajectory_index], max_scale_error)

        while not rospy.is_shutdown() and (rospy.get_time() - initial_time) < duration:

            current_wrench = self.get_wrench(base_frame_control=True)

            if termination_criteria is not None:
                assert isinstance(termination_criteria, types.LambdaType), "Invalid termination criteria, expecting lambda/function with one argument[current pose array[7]]"
                if termination_criteria(self.end_effector()):
                    rospy.loginfo("Termination criteria returned True, stopping force control")
                    result = ExecutionResult.TERMINATION_CRITERIA
                    break

            rospy.loginfo_throttle(1, 'F/T {}'.format(np.round(current_wrench[:3], 2)))
            if stop_on_target_force and is_more_extreme(current_wrench[stop_target_wrench_mask], stop_at_wrench[stop_target_wrench_mask]):
                rospy.loginfo('Target F/T reached {}'.format(np.round(current_wrench, 2)) + ' Stopping!')
                result = ExecutionResult.STOP_ON_TARGET_FORCE
                break

            # Safety limits: max force
            if np.any(np.abs(current_wrench) > max_force_torque):
                rospy.logerr('Maximum force/torque exceeded {}'.format(np.round(current_wrench, 3)))
                result = ExecutionResult.FORCE_TORQUE_EXCEEDED
                break

            if (rospy.get_time() - step_initial_time) > step_duration:
                step_initial_time = rospy.get_time()
                trajectory_index += 1
                if trajectory_index >= trajectory.shape[0]:
                    break
                # push next point to the controller
                self.set_cartesian_target_pose(trajectory[trajectory_index])

                if scale_up_error and max_scale_error:
                    self.sliding_error(trajectory[trajectory_index], max_scale_error)

            if func:
                func(self.end_effector())

            self.rate.sleep()

        if auto_stop:
            # Stop moving
            # set position control only, then fix the pose to the current one
            self.set_cartesian_target_pose(self.end_effector())
            self.set_position_control_mode()
            self.wait_for_robot_to_stop(wait_time=5)

        return result

    def sliding_error(self, target_pose, max_scale_error):
        # Scale error_scale as position error decreases until a max scale error
        position_error = np.linalg.norm(target_pose[:3] - self.end_effector()[:3])
        # from position_error < 0.01m increase scale error
        factor = 1 - np.tanh(100 * position_error)
        # scale_error = np.interp(factor, [0, 1], [0.01, max_scale_error])
        scale_error = np.interp(factor, [0, 1], [self.min_scale_error, max_scale_error])
        self.set_solver_parameters(error_scale=np.round(scale_error, 3))
