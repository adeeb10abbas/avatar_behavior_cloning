# Import some basic libraries and functions for this tutorial.
import numpy as np
from avatar_behavior_cloning.utils.geometry_utils import *

from pydrake.geometry import (
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
    StartMeshcat,
)
import argparse

from controllers.spatialstiffness_controller_avatar_arm import PoseControllerAvatar
from controllers.impedance_controller_avatar_hand import HandControllerAvatar

import pydot
import time

# to remove

num_positions = 7
panda_kp = [50] * num_positions
panda_ki = [1] * num_positions
panda_kd = [10] * num_positions
avatar_kp = []
avatar_kd = []


def arm_inv_dynamics_controller():
    inverse_dynamic_controller = InverseDynamicsController(
        plant,
        kp=panda_kp,  # 100
        ki=panda_ki,  # 1
        kd=panda_kd,  # 1
        has_reference_acceleration=False,
    )
    inv_dyn_franka_controller = builder.AddSystem(inverse_dynamic_controller)
    return inv_dyn_franka_controller


def arm_pose_controller():
    cartesian_stiffness_controller = PoseControllerAvatar(
        plant,
        franka,
        plant.GetFrameByName("panda_link0"),
        plant.GetFrameByName("panda_hand"),
    )
    cartesian_stiffness_controller = builder.AddSystem(cartesian_stiffness_controller)
    return cartesian_stiffness_controller


def hand_pid_controller():

    stiffness = 3.0
    state_projection_matrix = np.zeros((6, 20))
    index = [0, 2, 1]

    # joint / state:
    # right_thumb_flex_motor_joint
    # right_thumb_swivel_motor_joint
    # right_index_flex_motor_joint
    # right_thumb_knuckle_joint
    # right_thumb_finger_joint
    # right_thumb_swivel_joint
    # right_index_knuckle_joint
    # right_index_fingertip_joint
    # right_middle_knuckle_joint
    # right_middle_fingertip_joint

    # actuator:
    # right_thumb_flex_motor_joint
    # right_index_flex_motor_joint
    # right_thumb_swivel_motor_joint

    state_projection_matrix[0, 0] = 1
    state_projection_matrix[1, 2] = 1.0
    state_projection_matrix[2, 1] = 1.0

    # velocity
    state_projection_matrix[3, 10] = 1
    state_projection_matrix[4, 12] = 1.0
    state_projection_matrix[5, 11] = 1.0
    pid_controller = builder.AddSystem(
        PidController(
            state_projection=state_projection_matrix,
            kp=np.ones(hand_actuator_dim) * stiffness,
            ki=np.ones(hand_actuator_dim) * 0.01,
            kd=np.ones(hand_actuator_dim) * 0.01,
        )
    )  # np.ones
    return pid_controller


def hand_impedance_controller():
    cartesian_impedance_controller = HandControllerAvatar(
        plant=plant, model_instance=hand, joint_stiffness=[10, 10, 10]
    )
    cartesian_impedance_controller = builder.AddSystem(cartesian_impedance_controller)
    return cartesian_impedance_controller


# MACRO
hand_joint_dim = 3
hand_actuator_dim = 3
TIME_STEP = 0.001
ARM_CONTROLLER_TYPE = "pose"  # inv_dyn
ARM_STATE = "reset"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestep", type=float, default=0.001)
    parser.add_argument("--hand_controller_type", type=str, default="impedance")
    parser.add_argument("--hand_state", type=str, default="open")
    parser.add_argument("--motion", type=str, default="stable")
    parser.add_argument("--teleop_type", type=str, default="keyboard")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    TIME_STEP = args.timestep
    HAND_CONTROLLER_TYPE = args.hand_controller_type
    HAND_STATE = args.hand_state
    MOTION = args.motion

    #############################################
    # Initialize
    meshcat = StartMeshcat()
    builder = DiagramBuilder()
    multibody_plant_config = MultibodyPlantConfig(time_step=TIME_STEP, discrete_contact_solver="sap")
    plant, scene_graph = AddMultibodyPlant(multibody_plant_config, builder)

    #############################################
    # Add Plant


    parser = Parser(plant)
    # avatar_path = "data/avatar/urdf/avatar_gripper_3f.urdf"
    #  "models/avatar_gripper_3f.urdf"

    hand = parser.AddModelFromFile("data/avatar/urdf/avatar_gripper_3f.urdf")
    franka_combined_path = "data/avatar/urdf/panda_arm.urdf"
    franka = parser.AddModelFromFile(franka_combined_path)

    # set default joints
    q0 = [0.0] * 10
    index = 0
    for joint_index in plant.GetJointIndices(hand):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1

    q0 = [0.0, -1.285, 0, -2.356, 0.0, 1.571, 0.785]
    index = 0
    for joint_index in plant.GetJointIndices(franka):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1

    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0"))
    plant.WeldFrames(
        plant.GetFrameByName("panda_link8"),
        plant.GetFrameByName("panda_hand"),
        RigidTransform(RollPitchYaw(np.pi / 2, 0, 0), [0.0, 0, 0.0]),
    )
    if args.debug:
        simplify_plant(plant, scene_graph)

    plant.Finalize()

    print(f"num of joints: {plant.num_positions()} num of actuators: {plant.num_actuators()}")

    #############################################
    # Add Controller and connect diagram
    if ARM_CONTROLLER_TYPE == "inv_dyn":
        # This would fail as the plant contains more than the arm
        franka_controller = arm_inv_dynamics_controller()
        panda_positions = builder.AddSystem(PassThrough([0] * num_positions))
        desired_state_from_position = builder.AddSystem(
            StateInterpolatorWithDiscreteDerivative(num_positions, TIME_STEP, suppress_initial_transient=True)
        )
        builder.Connect(
            desired_state_from_position.get_output_port(),
            franka_controller.get_input_port_desired_state(),
        )

        builder.Connect(
            panda_positions.get_output_port(),
            franka_controller.get_input_port(),
        )
        builder.Connect(plant.get_state_output_port(franka), franka_controller.get_input_port_estimated_state())

        builder.ExportInput(panda_positions.get_input_port(), "panda_joint_commanded")
        arm_state_input_port = panda_positions.get_input_port()

    elif ARM_CONTROLLER_TYPE == "pose":
        franka_controller = arm_pose_controller()
        desired_eff_pose = builder.AddSystem(PassThrough(AbstractValue.Make(RigidTransform())))
        desired_eff_velocity = builder.AddSystem(PassThrough(AbstractValue.Make(SpatialVelocity())))
        builder.Connect(
            plant.get_state_output_port(franka),
            franka_controller.get_plant_state_input_port(),
        )

        builder.Connect(
            desired_eff_pose.get_output_port(),
            franka_controller.get_state_desired_port(),
        )
        builder.Connect(
            desired_eff_velocity.get_output_port(),
            franka_controller.get_velocity_desired_port(),
        )

        builder.ExportInput(desired_eff_pose.get_input_port(), "desired_eff_pose")
        builder.ExportInput(desired_eff_velocity.get_input_port(), "desired_eff_velocity")

        adder = builder.AddSystem(Adder(2, franka_controller.nu))
        torque_passthrough = builder.AddSystem(PassThrough([0] * franka_controller.nu))
        builder.Connect(franka_controller.get_output_port(), adder.get_input_port(0))
        builder.Connect(torque_passthrough.get_output_port(), adder.get_input_port(1))
        builder.ExportInput(torque_passthrough.get_input_port(), "feedforward_torque")
        builder.Connect(adder.get_output_port(), plant.get_actuation_input_port(franka))

    if HAND_CONTROLLER_TYPE == "pid":
        hand_controller = hand_pid_controller()
        builder.Connect(hand_controller.get_output_port_control(), plant.get_actuation_input_port(hand))
        passthrough_state = builder.AddSystem(PassThrough([0] * hand_actuator_dim))
        builder.Connect(plant.get_state_output_port(hand), hand_controller.get_input_port_estimated_state())
        desired_state_from_position_hand = builder.AddSystem(
            StateInterpolatorWithDiscreteDerivative(hand_actuator_dim, TIME_STEP, suppress_initial_transient=True)
        )
        builder.Connect(passthrough_state.get_output_port(), desired_state_from_position_hand.get_input_port())
        builder.Connect(
            desired_state_from_position_hand.get_output_port(), hand_controller.get_input_port_desired_state()
        )
        builder.ExportInput(passthrough_state.get_input_port(), "q_robot_commanded")

    elif HAND_CONTROLLER_TYPE == "impedance":
        hand_controller = hand_impedance_controller()
        builder.Connect(hand_controller.get_output_port_control(), plant.get_actuation_input_port(hand))
        passthrough_state = builder.AddSystem(PassThrough([0] * hand_joint_dim))
        builder.Connect(plant.get_state_output_port(hand), hand_controller.get_input_port_estimated_state())

        control_passthrough = builder.AddSystem(PassThrough([0] * hand_actuator_dim))
        builder.Connect(control_passthrough.get_output_port(), hand_controller.get_input_port_desired_state())
        builder.ExportInput(control_passthrough.get_input_port(), "q_robot_commanded")

        torque_passthrough = builder.AddSystem(PassThrough([0] * hand_joint_dim))
        builder.Connect(torque_passthrough.get_output_port(), hand_controller.get_torque_feedforward_port_control())
        builder.ExportInput(torque_passthrough.get_input_port(), "hand_feedforward_torque")

    #############################################
    visualizer = MeshcatVisualizer.AddToBuilder(
        builder,
        scene_graph,
        meshcat,  # kIllustration, kProximity
        MeshcatVisualizerParams(role=Role.kProximity, delete_on_initialization_event=False),
    )

    # Add contact visualizer
    cparams = ContactVisualizerParams()
    cparams.force_threshold = 1e-2
    cparams.newtons_per_meter = 1e6  # .0 # 1.0
    cparams.newton_meters_per_meter = 1e1
    cparams.radius = 0.005  # 0.01
    contact_visualizer = ContactVisualizer.AddToBuilder(builder, plant, meshcat, cparams)

    # Build the diagram
    diagram = builder.Build()
    hand_state_input_port = diagram.GetInputPort("q_robot_commanded")

    if ARM_CONTROLLER_TYPE == "pose":
        desired_eff_pose_port = diagram.GetInputPort("desired_eff_pose")
        desired_eff_vel_port = diagram.GetInputPort("desired_eff_velocity")

    # Set the initial conditions, x(0).
    context = diagram.CreateDefaultContext()
    pydot.graph_from_dot_data(diagram.GetGraphvizString())[0].write_svg("diagram.svg")

    #############################################
    # Create the simulator, and simulate for 10 seconds.
    simulator = Simulator(diagram, context)

    context = simulator.get_context()

    if MOTION == "stable":
        if HAND_STATE == "open":
            hand_state_input_port.FixValue(context, [0.0, 0.0, 0.0])
        else:
            hand_state_input_port.FixValue(context, [-1.0, -2.0, -1.0])

        if ARM_STATE == "reset":
            if ARM_CONTROLLER_TYPE == "inv_dyn":
                init_joints = [0.0, -1.285, 0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]
                arm_state_input_port.FixValue(context, init_joints)

            if ARM_CONTROLLER_TYPE == "pose":
                desired_eff_pose_port.FixValue(
                    context,
                    RigidTransform(
                        RollPitchYaw(-1.2080553034387211, -0.3528716676941929, -0.847106272058664),
                        [1.460e-01, 0, 7.061e-01],
                    ),
                )
                desired_eff_vel_port.FixValue(context, SpatialVelocity(np.zeros(6)))

        # check stableness
        simulator.AdvanceTo(100000)

    else:
        # TODO(lirui): create another script
        from teleop_utils import get_keyboard_input, get_vr_input

        print("use keyboard teleop: use w,s,a,d,q,e for arm and m,n for fingers")

        t = 0
        dt = 0.05

        desired_eff_vel_port.FixValue(context, SpatialVelocity(np.zeros(6)))
        endeffector_rot = np.array([-1.2080553034387211, -0.3528716676941929, -0.847106272058664])
        endeffector_trans = np.array([1.460e-01, 0, 7.061e-01])
        joint_pos_lower_limit = np.array([-1.0, -2.0, -1.0])
        joint_pos_upper_limit = np.array([0, 0, 0])
        finger_states = (joint_pos_lower_limit + joint_pos_upper_limit) / 2
        if args.teleop_type == "vr":
            # initialize the goal gripper
            vis_hand_pose(meshcat, np.eye(4), "hand_goal_pose", load=True)

        for _ in range(10000):
            if args.teleop_type == "keyboard":
                action, finger = get_keyboard_input()
                endeffector_trans += action[:3]
                endeffector_rot += action[3:]
            else:
                action, finger = get_vr_input()
                endeffector_trans[:] = action[:3]
                endeffector_rot[:] = action[3:]
                vis_hand_pose(meshcat, unpack_action(action), "hand_goal_pose")

            desired_eff_pose_port.FixValue(
                context,
                RigidTransform(RollPitchYaw(*endeffector_rot), endeffector_trans),
            )

            # could switch to individual finger as well
            if finger == "open":
                finger_states += 0.01, 0.01, 0.05
            elif finger == "close":
                finger_states -= 0.01, 0.01, 0.05

            finger_states = np.minimum(finger_states, joint_pos_upper_limit)
            finger_states = np.maximum(finger_states, joint_pos_lower_limit)

            # print("finger state:", finger_states)
            hand_state_input_port.FixValue(context, finger_states)

            # step simulator
            t = t + dt
            simulator.AdvanceTo(t)
            # time.sleep(1)