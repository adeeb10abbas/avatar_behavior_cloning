import numpy as np

from pydrake.systems.framework import (
    BasicVector,
    LeafSystem,
)
from pydrake.systems.framework import LeafSystem, BasicVector, PortDataType
from pydrake.multibody.tree import MultibodyForces, JacobianWrtVariable
from spatialstiffness_controller_avatar_arm import *
from collections import OrderedDict

# from drake_utils import get_joints, get_joint_actuators
def get_joints(plant, model_instances=None):

    # TODO(eric.cousineau): Hoist this somewhere?

    return _get_plant_aggregate(plant.num_joints, plant.get_joint, JointIndex, model_instances)


def get_joint_actuators(plant, model_instances=None):

    # TODO(eric.cousineau): Hoist this somewhere?

    return _get_plant_aggregate(plant.num_actuators, plant.get_joint_actuator, JointActuatorIndex, model_instances)


def _get_plant_aggregate(num_func, get_func, index_cls, model_instances=None):

    items = []

    for i in range(num_func()):

        item = get_func(index_cls(i))

        if model_instances is None or item.model_instance() in model_instances:

            items.append(item)

    return items
np.set_printoptions(3)
def get_joints(plant, model_instances=None):

    # TODO(eric.cousineau): Hoist this somewhere?

    return _get_plant_aggregate(plant.num_joints, plant.get_joint, JointIndex, model_instances)


def get_joint_actuators(plant, model_instances=None):

    # TODO(eric.cousineau): Hoist this somewhere?

    return _get_plant_aggregate(plant.num_actuators, plant.get_joint_actuator, JointActuatorIndex, model_instances)


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
        self, plant, model_instance, joint_stiffness, damping_ratio=1, controller_mode="impedance", debug_plot=False
    ):
        """
        Inverse dynamics controller makes use of drake's
            InverseDynamicsController. The version without mimic joints.
        :param plant_robot:
        :param joint_stiffness: (nq,) numpy array which defines the stiffness
            of all joints.

        joint / state:
        right_thumb_flex_motor_joint
        right_thumb_swivel_motor_joint
        right_index_flex_motor_joint
        right_thumb_knuckle_joint
        right_thumb_finger_joint
        right_thumb_swivel_joint
        right_index_knuckle_joint
        right_index_fingertip_joint
        right_middle_knuckle_joint
        right_middle_fingertip_joint

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
        self.joint_infos["right_thumb_flex_motor_joint"] = ["right_thumb_flex_motor_joint", 1, 0]
        self.joint_infos["right_thumb_swivel_motor_joint"] = ["right_thumb_swivel_motor_joint", 1, 0]
        self.joint_infos["right_index_flex_motor_joint"] = ["right_index_flex_motor_joint", 1, 0]
        self.joint_infos["right_thumb_knuckle_joint"] = ["right_thumb_flex_motor_joint", 0.5702, 0]
        self.joint_infos["right_thumb_finger_joint"] = ["right_thumb_flex_motor_joint", 0.6007, 0]
        self.joint_infos["right_thumb_swivel_joint"] = ["right_thumb_swivel_motor_joint", -0.6075, 0.43]
        self.joint_infos["right_index_knuckle_joint"] = ["right_index_flex_motor_joint", 1, 0]
        self.joint_infos["right_index_fingertip_joint"] = ["right_index_flex_motor_joint", 0.40, 0]
        self.joint_infos["right_middle_knuckle_joint"] = ["right_index_flex_motor_joint", 1, 0]
        self.joint_infos["right_middle_fingertip_joint"] = ["right_index_flex_motor_joint", 0.40, 0]

        self.actuator_index = [0, 1, 2]
        self.plant = plant
        self.controller_mode = controller_mode
        self.model_instance = model_instance

        context = plant.CreateDefaultContext()
        self.context = plant.CreateDefaultContext()

        self.nq = plant.num_positions(model_instance)
        self.nv = plant.num_velocities(model_instance)
        self.nu = 3  # hard-coded to debug plant.num_actuated_dofs(model_instance)
        self.nx = 2 * self.nq

        self.robot_state_input_port = self.DeclareInputPort(
            "robot_state", PortDataType.kVectorValued, self.nq + self.nv
        )
        self.tau_feedforward_input_port = self.DeclareInputPort("tau_feedforward", PortDataType.kVectorValued, self.nu)
        self.joint_angle_commanded_input_port = self.DeclareInputPort(
            "q_robot_commanded", PortDataType.kVectorValued, self.nu
        )
        self.joint_torque_output_port = self.DeclareVectorOutputPort(
            "joint_torques", BasicVector(self.nu), self.CalcJointTorques
        )

        # control rate
        self.t = 0
        self.control_period = 1e-3  # 1000Hz.
        self.DeclareDiscreteState(self.nu)
        self.DeclarePeriodicDiscreteUpdateNoHandler(self.control_period)

        # joint velocity estimator
        self.q_prev = None
        self.w_cutoff = 2 * np.pi * 4000
        self.velocity_estimator = LowPassFilter(self.nv, self.control_period, self.w_cutoff)
        self.acceleration_estimator = LowPassFilter(self.nv, self.control_period, self.w_cutoff)
        self.dofs = articulated_model_instance_dofset(plant, model_instance)

        # damping coefficient filter
        self.Kv_filter = LowPassFilter(self.nu, self.control_period, 2 * np.pi)

        # controller gains
        self.get_full_stiffness()  # joint_stiffness
        self.damping_ratio = damping_ratio
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

        # if debug_plot:

    def get_full_stiffness(self):
        self.Kp = np.array([10, 10, 10])  # , 10, 10, 10, 10, 10, 10, 10])  # np.ones(10)  #
        self.Kv = np.array([2, 2, 2])  # , 2, 2, 0.1, 2, 2, 2, 2])  # np.ones(10) * 2

    def expand_joints(self, target_actuator_joint_pos, no_bias=False):
        """
        enforce mimic joint constraints of the robot hand.
        compute the desired joints for all joints of the hand
        """
        target_actuator_joint_pos = np.minimum(target_actuator_joint_pos, self.joint_pos_upper_limit)
        target_actuator_joint_pos = np.maximum(target_actuator_joint_pos, self.joint_pos_lower_limit)

        target_actuator_dict = {
            "right_thumb_flex_motor_joint": target_actuator_joint_pos[0],
            "right_thumb_swivel_motor_joint": target_actuator_joint_pos[1],
            "right_index_flex_motor_joint": target_actuator_joint_pos[2],
        }
        target_joints = []
        for joint_name, (mimic_joint_name, w, b) in self.joint_infos.items():
            target_joint = target_actuator_dict[mimic_joint_name] * w
            target_joint += b
            target_joints.append(target_joint)
        # print("expanded target joints:", np.round(target_joints,2))
        return target_joints

    def CalcJointTorques(self, context, y_data):
        state = context.get_discrete_state_vector().get_value()
        y = y_data.get_mutable_value()
        # scale the joint torque as well
        y[:] = state[self.actuator_index]
        # y[:] = self.expand_joints(state, no_bias=True)  # [state[idx] for idx in self.actuator_index]

    def DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        LeafSystem.DoCalcDiscreteVariableUpdates(self, context, events, discrete_state)

        # read input ports
        x = self.robot_state_input_port.Eval(context)
        q_cmd = self.joint_angle_commanded_input_port.Eval(context)

        # conversion
        q_cmd = self.expand_joints(q_cmd)

        tau_ff = self.tau_feedforward_input_port.Eval(context)
        # import pdb; pdb.set_trace()
        q = x[: self.nq]
        v = x[self.nq :]

        # estimate velocity
        if self.q_prev is None:
            self.velocity_estimator.update(np.zeros(self.nv))
        else:
            # low pass filter velocity.
            v_diff = (q - self.q_prev) / self.control_period
            self.velocity_estimator.update(v_diff)

        self.q_prev = q
        v_est = self.velocity_estimator.get_current_state()

        # log the P and D parts of desired acceleration
        self.sample_times.append(context.get_time())

        # update plant context
        self.plant.SetPositions(self.context, self.model_instance, q)
        # self.plant.SetVelocities(self.context, v_est)

        # gravity compenstation
        tau_g = self.plant.CalcGravityGeneralizedForces(self.context)
        # import IPython; IPython.embed()
        tau = -tau_g[self.dofs.q][self.actuator_index]

        if self.controller_mode == "impedance":

            M = self.plant.CalcMassMatrixViaInverseDynamics(self.context)
            M = M[self.dofs.q][:, self.dofs.q]
            M = M[self.actuator_index][:, self.actuator_index]

            m = np.sort(np.linalg.eig(M)[0])[::-1]
            m = M.diagonal()
            Kv = 2 * self.damping_ratio * np.sqrt(self.Kp * m)
            self.Kv_filter.update(Kv)
            Kv = self.Kv_filter.get_current_state()

            tau_stiffness = self.Kp * (np.array(q_cmd)[self.actuator_index] - np.array(q)[self.actuator_index])
            tau_damping = -Kv * np.array(v)[self.actuator_index]
            tau += tau_damping + tau_stiffness

            # self.Kv_log.append(Kv)
            self.tau_stiffness_log.append(tau_stiffness.copy())
            self.tau_damping_log.append(tau_damping.copy())

            debug_j = 1
            if np.abs(v[debug_j]) > 0:
                # import IPython; IPython.embed()
                print(
                    f"debug joint: v: {v[debug_j]:.3f} q: {q[debug_j]:.3f} desired_q: {q_cmd[debug_j]:.3f} torque stiffness: {tau_stiffness[debug_j]:.3f}"
                )
                # q: -1.000 desired_q: -1.010 torque stiffness: -0.099
                # q: -1.000 desired_q: -1.000 torque stiffness: 0.001
                # q is constant
                # print(q_cmd[:3] - q[:3])

            # import pdf; pdb.set_trace()

        elif self.controller_mode == "inverse_dynamics":
            # compute desired acceleration
            qDDt_d = self.Kp * (q_cmd - q) + self.Kv * (-v_est)
            tau += self.plant.CalcInverseDynamics(
                context=self.context, known_vdot=qDDt_d, external_forces=MultibodyForces(self.plant)
            )

        output = discrete_state.get_mutable_vector().get_mutable_value()
        output[:] = tau[self.actuator_index] + tau_ff

    def get_input_port_estimated_state(self):
        return self.robot_state_input_port

    def get_input_port_desired_state(self):
        return self.joint_angle_commanded_input_port

    def get_output_port_control(self):
        return self.joint_torque_output_port

    def get_torque_feedforward_port_control(self):
        return self.tau_feedforward_input_port