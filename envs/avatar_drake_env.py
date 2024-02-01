# Standard library imports
import threading

# Local imports
from avatar_behavior_cloning.controllers.impedance_controller_avatar_hand import *
from avatar_behavior_cloning.utils.geometry_utils import *
from avatar_behavior_cloning.utils.teleop_utils import *
from avatar_behavior_cloning.utils.drake_utils import *
from avatar_behavior_cloning.nodes.ros_teleop_data_receiver_py import RosTeleopDataReceiver

# Third-party imports
from pydrake.geometry import MeshcatVisualizer, MeshcatVisualizerParams, Role, StartMeshcat
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlant, MultibodyPlant, MultibodyPlantConfig
from pydrake.multibody.tree import SpatialInertia
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
import numpy as np
import rclpy
import scipy

class AvatarDrakeEnv:
  def __init__(self, timestep, hand_controller_type, hand_state, arm_controller_type, motion, teleop_type, debug, plot):
    rclpy.init(args=None)
    self.ros_teleop_data_receiver_left = RosTeleopDataReceiver(side = "left")
    self.ros_teleop_data_receiver_right = RosTeleopDataReceiver(side = "right")
    self.executor = rclpy.executors.MultiThreadedExecutor()
    self.executor.add_node(self.ros_teleop_data_receiver_left)
    self.executor.add_node(self.ros_teleop_data_receiver_right)
    self.executor_thread = threading.Thread(target=self.executor.spin)
    self.executor_thread.start()
    ############################################# 
    # Set up parameters
    self.timestep = timestep
    self.motion = "arm_teleop"
    self.teleop_type = teleop_type
    self.debug = debug
    self.plot = plot

    self.arm_controller_type = arm_controller_type
    self.arm_state = {"zero": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                      "left_home": [0.1, 0.1, 0.0, -2.7, 1.5708, 1.5708, -2.0],
                      "right_home": [1.1, 0.1, 0.0, -2.7, 1.5708, 1.5708, -2.0],
                      "q1":   [0.1, 0.1, -0.5, -2.356, 0.0, 0, 0.785],
                      "q2":   [1.0, 0.5, -1.1, -2.356, 0.0, 0, 0.785]}
    self.panda_joints = 7
    self.panda_kp = [5000] * self.panda_joints
    self.panda_ki = [1] *  self.panda_joints
    self.panda_kd = [100] * self.panda_joints

    # Change this to 3 if using mimic
    self.hand_controller_type = hand_controller_type
    self.hand_state = hand_state
    self.hand_joint_dim = 10
    self.hand_actuator_dim = 3
    self.hand_kp = []
    self.hand_kd = []
    
    ## URDF Paths
    self.right_panda_arm_urdf = "data/avatar/urdf/avatar_right_arm_new.urdf"
    # self.right_hand_model_urdf = "data/avatar/urdf/avatar_gripper_right.urdf"

    self.left_panda_arm_urdf = "data/avatar/urdf/avatar_left_arm_new.urdf"
    self.left_hand_model_urdf = "data/avatar/urdf/avatar_gripper_left_nomimic.urdf"

    # self.avatar_base_urdf = "data/avatar/urdf/avatar_base.urdf"

    self.right_hand_model_urdf  = "data/avatar/urdf/avatar_gripper_right_nomimic.urdf"
    self.table_top_sdf = "data/avatar/env_objects/table_top.sdf"
  
    #############################################
    # Initialize
    self.meshcat = StartMeshcat()
    self.builder = DiagramBuilder()
    self.multibody_plant_config = MultibodyPlantConfig(time_step=self.timestep, discrete_contact_solver="sap")
    self.plant, self.scene_graph = AddMultibodyPlant(config=self.multibody_plant_config, builder=self.builder)
    self.parser = Parser(self.plant)
    # separate plant for controller
    self.left_controller_plant = MultibodyPlant(time_step=self.timestep)
    self.right_controller_plant = MultibodyPlant(time_step=self.timestep)
    self.left_hand_controller_plant = MultibodyPlant(time_step=self.timestep)
    self.right_hand_controller_plant = MultibodyPlant(time_step=self.timestep)

    self.left_controller_parser = Parser(self.left_controller_plant)
    self.right_controller_parser = Parser(self.right_controller_plant)

    self.env_plant = MultibodyPlant(time_step=self.timestep)
    self.env_parser = Parser(self.env_plant)
    
    self.left_controller_plant = MultibodyPlant(time_step=self.timestep)
    self.left_controller_parser = Parser(self.left_controller_plant)
    self.left_hand_controller_parser = Parser(self.left_hand_controller_plant)
    self.left_controller_model = None
    self.left_hand_controller_model = None

    self.right_controller_plant = MultibodyPlant(time_step=self.timestep)
    self.right_controller_parser = Parser(self.right_controller_plant)
    self.right_hand_controller_parser = Parser(self.right_hand_controller_plant)
    self.right_controller_model = None
    self.right_hand_controller_model = None
    
    self.hand_joint_lower_limit = []
    self.hand_joint_upper_limit = []
    
    self.register_whole_plant()
    self.register_panda_controller_plant()
    self.register_hand_controller_plant()
    # self.create_scene()

    #############################################
    # Add arm and hand controllers and connect diagram
    self.register_arm_controllers()
    self.register_hand_controllers("left")
    self.register_hand_controllers("right")
    #############################################
    # Arm and gripper states
    self.desired_eff_pose = np.zeros(7)
    self.desired_finger_pos = np.zeros(3)
  
    #############################################
    visualizer = MeshcatVisualizer.AddToBuilder(
        self.builder,
        self.scene_graph,
        self.meshcat,  # kIllustration, kProximity
        MeshcatVisualizerParams(role=Role.kIllustration, delete_on_initialization_event=False),
    )

    # Add contact visualizer
    cparams = ContactVisualizerParams()
    cparams.force_threshold = 1e-2
    cparams.newtons_per_meter = 1e6  # .0 # 1.0
    cparams.newton_meters_per_meter = 1e1
    cparams.radius = 0.005  # 0.01
    contact_visualizer = ContactVisualizer.AddToBuilder(self.builder, self.plant, self.meshcat, cparams)

    if self.plot:
        self.plotter = start_logger()

    # Build the diagram
    self.diagram = self.builder.Build()
    self.context = self.diagram.CreateDefaultContext()
    self.meshcat.AddButton("Stop Simulation", "Escape")
    self.meshcat.AddButton("Next Step", "Enter")
  
  def close(self):
        # Cleanup ROS-related resources
        self.executor.shutdown()
        rclpy.shutdown()

  def create_scene(self):
    # pass
    self.table_instance = self.parser.AddModelFromFile(self.table_top_sdf)
    self.plant.WeldFrames(
      self.plant.world_frame(),
      self.plant.GetFrameByName("table_top_center"),
      RigidTransform(RollPitchYaw(0,0,0), [0.5, 0., 0.5])
    )
  
  def register_whole_plant(self):
    """ Registers the whole plant for visualization
    """

    self.left_panda_arm_instance = self.parser.AddModelFromFile(self.left_panda_arm_urdf)
    self.right_panda_arm_instance = self.parser.AddModelFromFile(self.right_panda_arm_urdf)

    self.left_gripper_instance = self.parser.AddModelFromFile(self.left_hand_model_urdf)
    self.right_gripper_instance = self.parser.AddModelFromFile(self.right_hand_model_urdf)
    self.create_scene()
    self.init_joints()

    ## Weld everything together and visualize! 
    # self.plant.WeldFrames(self.plant.world_frame(), self.plant.GetFrameByName("base"),
    #                 RigidTransform(RollPitchYaw(0,0,0), [0.5, 0.2, 0.0]))

    ## Left Arm + Hand
    self.plant.WeldFrames(self.plant.world_frame(), self.plant.GetFrameByName("left_panda_link0"),
                    RigidTransform(RollPitchYaw(0,0,0), [0.06975, 0.27, 0.5]))

    self.plant.WeldFrames(self.plant.GetFrameByName("left_panda_link8"), self.plant.GetFrameByName("left_gripper_base"),
                    RigidTransform(RollPitchYaw(0, 3.141592653589793, -0.7853981633974483), [0.0, 0.0, 0.0]))
    ## Right Arm + Hand
    self.plant.WeldFrames(self.plant.world_frame(), self.plant.GetFrameByName("right_panda_link0"),
                    RigidTransform(RollPitchYaw(0,0,0), [0.06975, -0.27, 0.5]))

    self.plant.WeldFrames(self.plant.GetFrameByName("right_panda_link8"), self.plant.GetFrameByName("right_gripper_base"),
                        RigidTransform(RollPitchYaw(0, 3.141592653589793, -0.7853981633974483), [0.0, 0.0, 0.0]))
                    # RigidTransform(RollPitchYaw(1.57, 0, 0), [0.00736, 0.139, -0.066017]))
    self.plant.Finalize()

    plant_context = self.plant.CreateDefaultContext()
    print("plant q size: ", self.plant.num_positions())
    print("plant v size: ", self.plant.num_velocities())
    print("plant u size: ", self.plant.num_actuators())
    print("plant x size: ", self.plant.num_positions() + self.plant.num_velocities())
    print("get the num actuated dofs: ", self.plant.num_actuated_dofs())

    X_WorldTable = self.plant.world_frame().CalcPoseInWorld(plant_context)
    X_TableCylinder = RigidTransform(
        RollPitchYaw(np.asarray([0, 0, 0]) * np.pi / 180), p=[0,-0.2,0.1])
    X_WorldCylinder = X_WorldTable.multiply(X_TableCylinder)
    print("plant q size: ", self.plant.num_positions())
  
  def register_panda_controller_plant(self): 
    """ Registers a plant specifically for the panda controller with attached rigib body that has the same inertia as 
            the hand
    """
    hand_spatial_inertial = SpatialInertia.SolidEllipsoidWithMass(2, 0.05, 0.15, 0.07)
    # Add model for left hand control
    self.left_controller_model = self.left_controller_parser.AddModelFromFile(self.left_panda_arm_urdf)
    self.left_controller_plant.WeldFrames(
        self.left_controller_plant.world_frame(), 
        self.left_controller_plant.GetFrameByName("left_panda_link0")
    )
    
    left_equivalent_body  = self.left_controller_plant.AddRigidBody(
        "left_hand_equivalent_body", 
        self.left_controller_model,
        hand_spatial_inertial
    )
    
    self.left_controller_plant.WeldFrames(
        self.left_controller_plant.GetFrameByName("left_panda_link8"),
        left_equivalent_body.body_frame(), 
        RigidTransform(RollPitchYaw(np.pi, 0, 3*np.pi/4), [0.0, 0, 0.0])
    )
    self.left_controller_plant.set_name('left_controller_plant')
    self.left_controller_plant.Finalize()
    # left hand's q is 7
    print("left_hand_controller q size: ", self.left_controller_plant.num_positions())

    # Add model for right hand control
    self.right_controller_model = self.right_controller_parser.AddModelFromFile(self.right_panda_arm_urdf)
    self.right_controller_plant.WeldFrames(
        self.right_controller_plant.world_frame(), 
        self.right_controller_plant.GetFrameByName("right_panda_link0")
    )
    
    right_equivalent_body  = self.right_controller_plant.AddRigidBody(
        "right_equivalent_body", 
        self.right_controller_model,
        hand_spatial_inertial
    )
    
    self.right_controller_plant.WeldFrames(
        self.right_controller_plant.GetFrameByName("right_panda_link8"),
        right_equivalent_body.body_frame(), 
        RigidTransform(RollPitchYaw(np.pi, 0, 3*np.pi/4), [0.0, 0, 0.0])
    )
    self.right_controller_plant.set_name('right_controller_plant')
    self.right_controller_plant.Finalize()
    # right hand's q is 7
    print("right_controller q size: ", self.right_controller_plant.num_positions())

  def register_hand_controller_plant(self):
    """ Registers a plant specifically for the hand controller without the Panda arm
    """
    self.right_hand_controller_model = self.right_hand_controller_parser.AddModelFromFile(self.right_hand_model_urdf)
    self.right_hand_controller_plant.WeldFrames(
        self.right_hand_controller_plant.world_frame(),
        self.right_hand_controller_plant.GetFrameByName("right_gripper_base"),
        RigidTransform(RollPitchYaw(np.pi, 0, 3*np.pi/4), [0.0, 0, 0.0]),
    )
    self.right_hand_controller_plant.set_name('right_hand_controller_plant')
    self.right_hand_controller_plant.Finalize()
    print("RIGHT hand_controller q size: ", self.right_hand_controller_plant.num_positions())
  
    self.left_hand_controller_model = self.left_hand_controller_parser.AddModelFromFile(self.left_hand_model_urdf)
    self.left_hand_controller_plant.WeldFrames(
        self.left_hand_controller_plant.world_frame(),
        self.left_hand_controller_plant.GetFrameByName("left_gripper_base"),
        RigidTransform(RollPitchYaw(np.pi, 0, 3*np.pi/4), [0.0, 0, 0.0]),
    )
    self.left_hand_controller_plant.set_name('left_hand_controller_plant')
    self.left_hand_controller_plant.Finalize()
    print("LEFT hand_controller q size: ", self.left_hand_controller_plant.num_positions())

  def set_default_angle_for_instance(self, instance, default_angles=None, is_gripper=False):
      """
      Sets the default angle for the joints of a given instance.
      If default angles are not provided, zeros are used for each joint.
      For gripper instances, updates the hand joint limits as well.
      """
      if default_angles is None:
          default_angles = [0] * self.hand_joint_dim
      index = 0
      for joint_index in self.plant.GetJointIndices(instance):
          joint = self.plant.get_mutable_joint(joint_index)
          if isinstance(joint, RevoluteJoint):
              joint.set_default_angle(default_angles[index])
              index += 1
              if is_gripper:
                  self.hand_joint_lower_limit.append(joint.position_lower_limit())
                  self.hand_joint_upper_limit.append(joint.position_upper_limit())

  def init_joints(self):
      # Initialize default angles for arm and hand joints
      self.set_default_angle_for_instance(self.left_panda_arm_instance, self.arm_state['left_home'])
      self.set_default_angle_for_instance(self.right_panda_arm_instance, self.arm_state['right_home'])
      self.set_default_angle_for_instance(self.left_gripper_instance, is_gripper=True)
      self.set_default_angle_for_instance(self.right_gripper_instance, is_gripper=True)

      print("DEFAULT ANGLES SET")
  
  def register_arm_controllers(self):
    ## Only pose right now. The others don't work afaik.
    left_panda_controller = self.builder.AddSystem(
        PoseControllerAvatar(
        self.plant, 
        self.left_panda_arm_instance, 
        self.plant.GetFrameByName("left_panda_link0"), 
        self.plant.GetFrameByName("left_gripper_base"))
    )
    left_desired_eff_pose = self.builder.AddSystem(PassThrough(AbstractValue.Make(RigidTransform())))
    left_desired_eff_velocity = self.builder.AddSystem(PassThrough(AbstractValue.Make(SpatialVelocity())))
    self.builder.Connect(
        self.plant.get_state_output_port(self.left_panda_arm_instance),
        left_panda_controller.get_plant_state_input_port(),
    )

    self.builder.Connect(
        left_desired_eff_pose.get_output_port(),
        left_panda_controller.get_state_desired_port(),
    )
    self.builder.Connect(
        left_desired_eff_velocity.get_output_port(),
        left_panda_controller.get_velocity_desired_port(),
    )

    self.builder.ExportInput(left_desired_eff_pose.get_input_port(), "left_desired_eff_pose")
    self.builder.ExportInput(left_desired_eff_velocity.get_input_port(), "left_desired_eff_velocity")

    left_adder = self.builder.AddSystem(Adder(2, left_panda_controller.nu))
    left_torque_passthrough = self.builder.AddSystem(PassThrough([0] * left_panda_controller.nu))
    self.builder.Connect(left_panda_controller.get_output_port(), left_adder.get_input_port(0))
    self.builder.Connect(left_torque_passthrough.get_output_port(), left_adder.get_input_port(1))
    self.builder.ExportInput(left_torque_passthrough.get_input_port(), "left_feedforward_torque")
    self.builder.Connect(left_adder.get_output_port(), self.plant.get_actuation_input_port(self.left_panda_arm_instance))   

    ## Export the left arm state 
    self.builder.ExportOutput(self.plant.get_state_output_port(self.left_panda_arm_instance), "left_panda_state")   
    
    right_panda_controller = self.builder.AddSystem(
        PoseControllerAvatar(
        self.plant, 
        self.right_panda_arm_instance, 
        self.plant.GetFrameByName("right_panda_link0"), 
        self.plant.GetFrameByName("right_gripper_base"))
    )
    right_desired_eff_pose = self.builder.AddSystem(PassThrough(AbstractValue.Make(RigidTransform())))
    right_desired_eff_velocity = self.builder.AddSystem(PassThrough(AbstractValue.Make(SpatialVelocity())))
    self.builder.Connect(
        self.plant.get_state_output_port(self.right_panda_arm_instance),
        right_panda_controller.get_plant_state_input_port(),
    )
    self.builder.Connect(
        right_desired_eff_pose.get_output_port(),
        right_panda_controller.get_state_desired_port(),
    )
    self.builder.Connect(
        right_desired_eff_velocity.get_output_port(),
        right_panda_controller.get_velocity_desired_port(),
    )

    self.builder.ExportInput(right_desired_eff_pose.get_input_port(), "right_desired_eff_pose")
    self.builder.ExportInput(right_desired_eff_velocity.get_input_port(), "right_desired_eff_velocity")

    right_adder = self.builder.AddSystem(Adder(2, right_panda_controller.nu))
    right_torque_passthrough = self.builder.AddSystem(PassThrough([0] * right_panda_controller.nu))
    self.builder.Connect(right_panda_controller.get_output_port(), right_adder.get_input_port(0))
    self.builder.Connect(right_torque_passthrough.get_output_port(), right_adder.get_input_port(1))
    self.builder.ExportInput(right_torque_passthrough.get_input_port(), "right_feedforward_torque")
    self.builder.Connect(right_adder.get_output_port(), self.plant.get_actuation_input_port(self.right_panda_arm_instance))      
    ## Export the right arm state
    self.builder.ExportOutput(self.plant.get_state_output_port(self.right_panda_arm_instance), "right_panda_state")

  def register_hand_controllers(self, hand_side: str):
    '''
    Registers the Hand Controllers and connects them to the plant. Only impedance control for now. 
    '''
    if hand_side == "right":
        gripper_instance = self.right_gripper_instance
        hand_commanded_input = "right_q_robot_commanded"
        hand_feedforward_torque_input = "right_hand_feedforward_torque"
        hand_controller_output = "right_hand_controller_output"
        hand_state_output = "right_hand_state"
    elif hand_side == "left":
        gripper_instance = self.left_gripper_instance
        hand_commanded_input = "left_q_robot_commanded"
        hand_feedforward_torque_input = "left_hand_feedforward_torque"
        hand_controller_output = "left_hand_controller_output"
        hand_state_output = "left_hand_state"
    else:
        raise ValueError("Invalid hand_side argument. Must be 'right' or 'left'.")

    hand_controller = self.builder.AddSystem(HandControllerAvatar(
        plant=self.plant,
        model_instance=gripper_instance,
        damping_ratio=1,
        hand=hand_side,
        controller_mode="impedance"))

    self.builder.Connect(hand_controller.get_output_port_control(), self.plant.get_actuation_input_port(gripper_instance))
    self.builder.Connect(self.plant.get_state_output_port(gripper_instance), hand_controller.get_input_port_estimated_state())

    control_passthrough = self.builder.AddSystem(PassThrough([0] * self.hand_actuator_dim))
    self.builder.Connect(control_passthrough.get_output_port(), hand_controller.get_input_port_desired_state())
    self.builder.ExportInput(control_passthrough.get_input_port(), hand_commanded_input)

    torque_passthrough = self.builder.AddSystem(PassThrough([0] * self.hand_joint_dim))
    self.builder.Connect(torque_passthrough.get_output_port(), hand_controller.get_torque_feedforward_port_control())
    self.builder.ExportInput(torque_passthrough.get_input_port(), hand_feedforward_torque_input)
    self.builder.ExportOutput(hand_controller.get_output_port_control(), hand_controller_output)
    self.builder.ExportOutput(self.plant.get_state_output_port(gripper_instance), hand_state_output)


  def hand_pid_controller(self):

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
    pid_controller = self.builder.AddSystem(
        PidController(
            state_projection=state_projection_matrix,
            kp=np.ones(self.hand_actuator_dim) * stiffness,
            ki=np.ones(self.hand_actuator_dim) * 0.01,
            kd=np.ones(self.hand_actuator_dim) * 0.01,
        )
    )  # np.ones
    return pid_controller

  def process_arm(self, arm_name, teleop_data_receiver, context, rot_init, joint_pos_lower_limit, joint_pos_upper_limit):
    desired_eff_pose = teleop_data_receiver.get_eff_pose(mode="simulated")
    desired_hand_state = teleop_data_receiver.get_finger_pos(mode="simulated")

    print(f"desired_{arm_name}_arm: ", desired_eff_pose)
    print(f"desired_{arm_name}_hand: ", desired_hand_state)

    endeffector_trans = desired_eff_pose[:3] + np.array([0.4, 0.1, 0.3])  # Adjust as needed

    rot_relative = scipy.spatial.transform.Rotation.from_quat(desired_eff_pose[3:])
    rot = rot_relative * rot_init
    rot_euler = rot.as_euler('xyz')
    endeffector_rot = rot_euler

    eff_pose_port = self.diagram.GetInputPort(f"{arm_name}_desired_eff_pose")
    hand_state_input_port = self.diagram.GetInputPort(f"{arm_name}_q_robot_commanded")

    # eff_pose_port = getattr(self, eff_pose_port_name)
    # hand_state_input_port = getattr(self, hand_state_input_port_name)

    eff_pose_port.FixValue(
        context,
        RigidTransform(RollPitchYaw(*endeffector_rot), endeffector_trans),
    )

    np.clip(desired_hand_state, joint_pos_lower_limit, joint_pos_upper_limit, out=desired_hand_state)
    hand_state_input_port.FixValue(context, desired_hand_state)

  def process_keyboard(self, arm_name, context, rot_init, joint_pos_lower_limit, joint_pos_upper_limit):
        ## Do this for both arms 
        action, finger = get_keyboard_input()
        # endeffector_trans += action[:3]
        # endeffector_rot += action[3:]

        # desired_eff_pose_port.FixValue(
        #     context,
        #     RigidTransform(RollPitchYaw(*endeffector_rot), endeffector_trans),
        # )

        # # could switch to individual finger as well
        # if finger == "thumb_up":
        #     finger_states += 0.00, 0.1, 0.00
        # elif finger == "thumb_down":
        #     finger_states -= 0.00, 0.1, 0.00
        # elif finger == "thumb_open":
        #     finger_states += 0.1, 0.00, 0.00
        # elif finger == "thumb_close":
        #     finger_states -= 0.1, 0.00, 0.00
        # elif finger == "index_open":
        #     finger_states += 0.00, 0.00, 0.1
        # elif finger == "index_close":
        #     finger_states -= 0.00, 0.00, 0.1

        # np.clip(finger_states, joint_pos_lower_limit, joint_pos_upper_limit, out=finger_states)
        # hand_state_input_port.FixValue(context, finger_states)

  def simulate(self):
    simulator = Simulator(self.diagram, self.context)

    context = simulator.get_context()
    left_hand_state_input_port = self.diagram.GetInputPort("left_q_robot_commanded")
    right_hand_state_input_port = self.diagram.GetInputPort("right_q_robot_commanded")


    finger_states = [0.0, 0.0, 0.0]

    simulator.set_target_realtime_rate(1)
    t = 0
    dt = 0.1

    left_desired_eff_pose_port = self.diagram.GetInputPort("left_desired_eff_pose")
    right_desired_eff_pose_port = self.diagram.GetInputPort("right_desired_eff_pose")

    left_desired_eff_vel_port  = self.diagram.GetInputPort("left_desired_eff_velocity")
    right_desired_eff_vel_port  = self.diagram.GetInputPort("right_desired_eff_velocity")

    left_desired_eff_vel_port.FixValue(self.context, SpatialVelocity(np.zeros(6)))
    right_desired_eff_vel_port.FixValue(self.context, SpatialVelocity(np.zeros(6)))
    ###  ### ### ### 
    endeffector_rot = np.array([1.57, 1.57, 0.0])
    endeffector_trans = np.array([0.4, -0.1, 0.3])
    joint_pos_lower_limit = np.array(self.hand_joint_lower_limit[7:10])
    joint_pos_upper_limit = np.array(self.hand_joint_upper_limit[7:10])
    finger_states = np.array([0.0, 0.0, 0.0])
    ## ## ## ### ###
    self.desired_eff_pose[6] = 1.0
    rot_init = scipy.spatial.transform.Rotation.from_euler('xyz', [1.57, 1.57, 0.0])

    if self.motion == 'arm_teleop':
        while rclpy.ok():
            self.process_arm("left", self.ros_teleop_data_receiver_left, context, rot_init, joint_pos_lower_limit, joint_pos_upper_limit)
            self.process_arm("right", self.ros_teleop_data_receiver_right, context, rot_init, joint_pos_lower_limit, joint_pos_upper_limit)
            
            ## Just testing stuff out
            # left_hand_state_input_port.FixValue(context, finger_states)
            # right_hand_state_input_port.FixValue(context, finger_states)
            ##
            # step simulator
            t = t + dt
            simulator.AdvanceTo(t)
    else:
        while self.meshcat.GetButtonClicks("Stop Simulation") < 1:

            self.process_keyboard("left", context, rot_init, joint_pos_lower_limit, joint_pos_upper_limit)
            self.process_keyboard("right", context, rot_init, joint_pos_lower_limit, joint_pos_upper_limit)

            # step simulator
            t = t + dt
            simulator.AdvanceTo(t)
       
    if self.plot:
        debug_joint = 1
        desired_joint = finger_states[debug_joint]
        plant_context = self.plant.GetMyMutableContextFromRoot(context)
        hand_joints = self.plant.GetPositions(plant_context, self.hand_instance)
        actual_joint  = hand_joints[debug_joint]
        update_line(self.plotter[0], self.plotter[1], [t, actual_joint], [t, desired_joint])
