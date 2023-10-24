import numpy as np

from avatar_behavior_cloning.utils.drake_utils import *
from avatar_behavior_cloning.controllers.spatialstiffness_controller_avatar_arm import *

from pydrake.systems.framework import (
    BasicVector,
    LeafSystem,
)
from pydrake.systems.framework import LeafSystem, BasicVector, PortDataType
from pydrake.multibody.tree import MultibodyForces, JacobianWrtVariable
from collections import OrderedDict

import numpy as np
import time
import numpy as np
from numpy.linalg import inv

from pydrake.common.cpp_param import List
from pydrake.common.value import Value
from pydrake.math import RigidTransform
from pydrake.multibody.math import SpatialForce, SpatialVelocity

from pydrake.multibody.tree import (
    Frame,
    JacobianWrtVariable,
)
from pydrake.systems.framework import (
    FixedInputPortValue,
    InputPort,
    LeafSystem,
)
import dataclasses as dc
from transforms3d.axangles import *


def rotation_matrix_to_axang3(rot_mat):
    axangle = mat2axangle(rot_mat.matrix()[:3, :3])
    return axangle[0] * axangle[1]


class LowPassFilter:
    def __init__(self, dimension: int, h: float, w_cutoff: float):
        if w_cutoff == np.inf:
            self.a = 1.0
        else:
            self.a = h * w_cutoff / (1 + h * w_cutoff)

        self.n = dimension
        self.x = None

    def update(self, u: np.array):
        assert u.size == self.n

        if self.x is None:
            self.x = u
        else:
            self.x = (1 - self.a) * self.x + self.a * u

    def reset_state(self):
        self.x = None

    def has_valid_state(self):
        return self.x is not None

    def get_current_state(self):
        assert self.x is not None
        return self.x.copy()


class HandControllerAvatar(LeafSystem):
    """
    Implement Joint Impedance Controller just for the hand
    """

    def __init__(
        self, plant, model_instance, hand, damping_ratio=1, controller_mode="impedance", debug_plot=False
    ):
        """
        Inverse dynamics controller makes use of drake's
            InverseDynamicsController. The version without mimic joints.
        :param plant_robot:
        :param joint_stiffness: (nq,) numpy array which defines the stiffness
            of all joints.

        joint / state:
        right_thumb_knuckle_joint
        right_thumb_finger_joint
        right_thumb_swivel_joint
        right_index_knuckle_joint
        right_index_fingertip_joint
        right_middle_knuckle_joint
        right_middle_fingertip_joint
        right_thumb_flex_motor_joint
        right_thumb_swivel_motor_joint
        right_index_flex_motor_joint

        actuator:
        right_thumb_flex_motor_joint
        right_thumb_swivel_motor_joint
        right_index_flex_motor_joint

        """
        super().__init__()

        # joints = get_joint_actuators(plant, [model_instance])
        # for joint in joints:
        #     print(joint.name())
        # import pdb; pdb.set_trace()

        # joint name: [mimic joint index, scale, offset]
        self.joint_infos = OrderedDict()
        self.joint_infos[f"{hand}_thumb_knuckle_joint"] = [f"{hand}_thumb_flex_motor_joint", 0.28, 0]
        self.joint_infos[f"{hand}_thumb_finger_joint"] = [f"{hand}_thumb_flex_motor_joint", 0.24, 0]
        self.joint_infos[f"{hand}_thumb_swivel_joint"] = [f"{hand}_thumb_swivel_motor_joint", -0.6075, 0.43]
        self.joint_infos[f"{hand}_index_knuckle_joint"] = [f"{hand}_index_flex_motor_joint", 0.4016, 0]
        self.joint_infos[f"{hand}_index_fingertip_joint"] = [f"{hand}_index_flex_motor_joint", 0.40, 0]
        self.joint_infos[f"{hand}_middle_knuckle_joint"] = [f"{hand}_index_flex_motor_joint", 0.4016, 0]
        self.joint_infos[f"{hand}_middle_fingertip_joint"] = [f"{hand}_index_flex_motor_joint", 0.40, 0]
        self.joint_infos[f"{hand}_thumb_flex_motor_joint"] = [f"{hand}_thumb_flex_motor_joint", 1, 0]
        self.joint_infos[f"{hand}_thumb_swivel_motor_joint"] = [f"{hand}_thumb_swivel_motor_joint", 1, 0]
        self.joint_infos[f"{hand}_index_flex_motor_joint"] = [f"{hand}_index_flex_motor_joint", 1, 0]

        self.actuator_index = [7, 8, 9]
        self.plant = plant
        self.controller_mode = controller_mode
        self.model_instance = model_instance
        self.hand = hand
        context = plant.CreateDefaultContext()
        self.context = plant.CreateDefaultContext()

        self.nq = plant.num_positions(model_instance)
        self.nv = plant.num_velocities(model_instance)

        self.nu = 3  # hard-coded to debug plant.num_actuated_dofs(model_instance)
        self.nx = 2 * self.nq

        self.robot_state_input_port = self.DeclareInputPort(
            "robot_state", PortDataType.kVectorValued, self.nq + self.nv
        )
        self.tau_feedforward_input_port = self.DeclareInputPort("tau_feedforward", PortDataType.kVectorValued, self.nv)
        self.joint_angle_commanded_input_port = self.DeclareInputPort(
            "q_robot_commanded", PortDataType.kVectorValued, self.nu
        )
        self.joint_torque_output_port = self.DeclareVectorOutputPort(
            "joint_torques", BasicVector(self.nv), self.CalcJointTorques
        )

        # control rate
        self.t = 0
        self.control_period = 0.001  # 1000Hz. Determines how fast the controller is being called
        self.DeclareDiscreteState(self.nv)
        self.DeclarePeriodicDiscreteUpdateNoHandler(self.control_period)

        # joint velocity estimator
        self.q_prev = None
        # Set to high frequency (no filter is needed for v)
        self.w_cutoff = 2 * np.pi * 1000
        self.velocity_estimator = LowPassFilter(self.nv, self.control_period, self.w_cutoff)
        self.dofs = articulated_model_instance_dofset(plant, model_instance)

        # controller gains
        self.damping_ratio = damping_ratio
        self.get_full_stiffness()  # joint_stiffness
        # self.Kv = 2 * self.damping_ratio * np.sqrt(self.Kp)

        # to define later
        self.joint_pos_lower_limit = np.array([-1.0, -2.0, -1.0])
        self.joint_pos_upper_limit = np.array([0, 0, 0])
        self.joint_vel_limit = np.array([2.0, 2.0, 2.0])
        self.joint_effort_limit = np.array([1000, 1000, 1000])

        self.Kv_log = []
        self.tau_stiffness_log = []
        self.tau_damping_log = []
        self.sample_times = []

        self.debug_count = 0
        # if debug_plot:

    def get_full_stiffness(self):
        print("nq: ---->>>>>>", self.nq)
        self.Kp = 20*np.array([1, 0.25, 1, 1, 0.25, 1, 0.25, 1, 1, 1])  # np.ones(10)  #
        # self.Kp = np.concatenate(self.Kp, np.ones(6))
        M = self.plant.CalcMassMatrixViaInverseDynamics(self.context)
        print("RAW M SHAPE---------------- ", M.shape)
        # assert self.dofs.v.shape == (10,10), f"Actual number of v size: {self.dofs.v}"
        print("DOFS.V SHAPE---------------- ", self.dofs.v.shape)
        print("DOFS.V Type ---------------- ", type(self.dofs.v))

        M = M[self.dofs.v][:, self.dofs.v] 
        # M = M[self.actuator_index]  # [:, self.actuator_index]

        m = np.sort(np.linalg.eig(M)[0])[::-1]
        m = np.array(M.diagonal())
        # For some reason, the order of the state output port is different from the order defined in urdf
        m[0:3] = [m[1], m[2], m[0]]
        
        print("Size of self.Kp:", self.Kp.shape) # (10,)
        print("Size of M:", M.shape) # (16, 16)
        print("Size of m:", m.shape) # (16,)
        
        self.Kv = 2 * self.damping_ratio * np.sqrt(self.Kp * m)
        # self.Kv = self.Kv[:10]

    def expand_joints(self, target_actuator_joint_pos, no_bias=False):
        """
        enforce mimic joint constraints of the robot hand.
        compute the desired joints for all joints of the hand
        """
        # target_actuator_joint_pos = np.minimum(target_actuator_joint_pos, self.joint_pos_upper_limit)
        # target_actuator_joint_pos = np.maximum(target_actuator_joint_pos, self.joint_pos_lower_limit)

        target_actuator_dict = {
            f"{self.hand}_thumb_flex_motor_joint": target_actuator_joint_pos[0],
            f"{self.hand}_thumb_swivel_motor_joint": target_actuator_joint_pos[1],
            f"{self.hand}_index_flex_motor_joint": target_actuator_joint_pos[2],
        }
        target_joints = []
        for joint_name, (mimic_joint_name, w, b) in self.joint_infos.items():
            # print(f"joint_name: {joint_name}, mimic_joint_name: {mimic_joint_name}")
            target_joint = target_actuator_dict[mimic_joint_name] * w
            target_joint += b
            target_joints.append(target_joint)
        return target_joints

    def CalcJointTorques(self, context, y_data):
        state = context.get_discrete_state_vector().get_value()
        y = y_data.get_mutable_value()
        # scale the joint torque as well
        y[:] = state
        # y[:] = self.expand_joints(state, no_bias=True)  # [state[idx] for idx in self.actuator_index]

    def DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        LeafSystem.DoCalcDiscreteVariableUpdates(self, context, events, discrete_state)

        # read input ports
        x = self.robot_state_input_port.Eval(context)
        q_cmd = self.joint_angle_commanded_input_port.Eval(context)

        # conversion
        q_cmd = self.expand_joints(q_cmd)

        tau_ff = self.tau_feedforward_input_port.Eval(context)
        q = x[: self.nq]
        # For some reason, the order of the state output port is different from the order defined in urdf
        q[0:3] = [q[1], q[2], q[0]]
        v = x[self.nq :]
        v[0:3] = [v[1], v[2], v[0]]

        # estimate velocity
        if self.q_prev is None:
            self.velocity_estimator.update(np.zeros(self.nv))
        else:
            # low pass filter velocity.
            v_diff = (q - self.q_prev) / self.control_period
            self.velocity_estimator.update(v_diff)

        self.q_prev = q
        # v_est = self.velocity_estimator.get_current_state()

        # log the P and D parts of desired acceleration
        self.sample_times.append(context.get_time())

        # update plant context
        self.plant.SetPositions(self.context, self.model_instance, q)
        # self.plant.SetVelocities(self.context, v_est)

        # gravity compenstation
        tau_g = self.plant.CalcGravityGeneralizedForces(self.context)

        tau = -tau_g[self.dofs.v]
        self.debug_count += 1

        if self.controller_mode == "impedance":
            # temp = np.array([q[2], q[0], q[1]])
            # q[0:3] = temp
            tau_stiffness = self.Kp * (q_cmd - q)
            tau_damping = -self.Kv * v
            tau += tau_damping + tau_stiffness

            # self.Kv_log.append(Kv)
            self.tau_stiffness_log.append(tau_stiffness.copy())
            self.tau_damping_log.append(tau_damping.copy())
            
            # print(f"debug count: {self.debug_count}")

            # for debug_j in range(self.nq):
            #     print(
            #         f"debug joint {debug_j}: q: {q[debug_j]:.3f} desired_q: {q_cmd[debug_j]:.3f} output torque: {tau[debug_j]:.3f}"
            #     )
            # q does not change seems to be the problem
            # q is constant
            # import pdf; pdb.set_trace()

        elif self.controller_mode == "inverse_dynamics":
            # compute desired acceleration
            qDDt_d = self.Kp * (q_cmd - q) + self.Kv * (-v)
            # print(qDDt_d)
            tau = self.plant.CalcInverseDynamics(
                context=self.context, known_vdot=qDDt_d, external_forces=MultibodyForces(self.plant)
            )
            # for debug_j in range(self.nq):
            #     print(
            #         f"debug joint {debug_j}: q: {q[debug_j]:.3f} desired_q: {q_cmd[debug_j]:.3f} output torque: {tau[debug_j]:.3f}"
            #     )

        output = discrete_state.get_mutable_vector().get_mutable_value()
        output[:] = tau

    def get_input_port_estimated_state(self):
        return self.robot_state_input_port

    def get_input_port_desired_state(self):
        return self.joint_angle_commanded_input_port

    def get_output_port_control(self):
        return self.joint_torque_output_port

    def get_torque_feedforward_port_control(self):
        return self.tau_feedforward_input_port