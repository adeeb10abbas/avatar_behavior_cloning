# 
import numpy as np
import threading
import graphviz

from avatar_behavior_cloning.controllers.impedance_controller_avatar_hand import *
from avatar_behavior_cloning.utils.geometry_utils import *
from avatar_behavior_cloning.utils.drake_utils import *
from avatar_behavior_cloning.nodes.ros_teleop_data_receiver_py import RosTeleopDataReceiver
from pydrake.geometry import (
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
    StartMeshcat,
)
import matplotlib.pyplot as plt
from pydrake.systems.controllers import JointStiffnessController, InverseDynamics
from pydrake.systems.drawing import plot_graphviz, plot_system_graphviz
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, OutputPort
from pydrake.multibody.plant import AddMultibodyPlant, MultibodyPlantConfig, MultibodyPlant
from pydrake.multibody.tree import SpatialInertia, ModelInstanceIndex

import pydot
import click
import time
import cairosvg
import matplotlib.pyplot as plt

import rclpy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
import scipy

class AvatarDrakeEnv:
  def __init__(self, timestep, hand_controller_type, hand_state, arm_controller_type, motion, teleop_type, debug, plot):
    rclpy.init(args=None)
    self.ros_teleop_data_receiver = RosTeleopDataReceiver()
    self.executor = rclpy.executors.MultiThreadedExecutor()
    self.executor.add_node(self.ros_teleop_data_receiver)
    ############################################# 
    # Set up parameters
    self.timestep = timestep
    self.motion = motion
    self.teleop_type = teleop_type
    self.debug = debug
    self.plot = plot

    self.arm_controller_type = arm_controller_type
    self.arm_state = {"zero": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                      "home": [0.1, 0.1, 0.0, -2.7, 1.5708, 1.5708, -2.0],
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
    self.register_hand_controllers()
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

    pydot.graph_from_dot_data(self.diagram.GetGraphvizString())[0].write_png('diagram_systems.png')

    top_string = self.plant.GetTopologyGraphvizString()
    py_dot_string = pydot.graph_from_dot_data(top_string)[0]
    py_dot_string.write_png('diagram_plant.png')
    # Save the graph as an image
    # top_string.write_png('diagram.png')

    # cairosvg.svg2png(url='diagram.svg').write_to_png('diagram.png')

    # self.panda_state = self.diagram.GetOutputPort("panda_state")

    # Start a separate thread to spin ros
    self.ros_spin_thread = threading.Thread(target=self.spin_ros)
    self.ros_spin_thread.start()

  def spin_ros(self):
      try:
          self.executor.spin()
      except KeyboardInterrupt:
          pass
  
  def update_from_ros(self):
      # Update the environment state with the latest data received from ROS
      if self.ros_teleop_data_receiver.desired_finger_pos is not None:
          self.desired_finger_pos = self.ros_teleop_data_receiver.desired_finger_pos

      if self.ros_teleop_data_receiver.desired_eff_pose is not None:
          self.desired_eff_pose = self.ros_teleop_data_receiver.desired_eff_pose
  
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

  def setup_manipulation_scene(self):
    """ Adds panda, hand, table and object to the scene
    """
    pass

  def setup_default_state(self):
    """ Sets up the default state for the chosen setup
    """
    pass
  
  def setup_random_state(self):
    """ Sets up a random state for the chosen setup
    """
    pass
  
  def register_whole_plant(self):
    """ Registers the whole plant for visualization
    """
    # Add model for visualization
    # self.panda_instance = self.parser.AddModelFromFile(self.panda_model_urdf)
    # self.hand_instance = self.parser.AddModelFromFile(self.hand_model_urdf)
    self.left_panda_arm_instance = self.parser.AddModelFromFile(self.left_panda_arm_urdf)
    self.right_panda_arm_instance = self.parser.AddModelFromFile(self.right_panda_arm_urdf)

    self.left_gripper_instance = self.parser.AddModelFromFile(self.left_hand_model_urdf)
    self.right_gripper_instance = self.parser.AddModelFromFile(self.right_hand_model_urdf)
    
    # self.avatar_base_instance = self.parser.AddModelFromFile(self.avatar_base_urdf)
    # self.table_instance = self.parser.AddModelFromFile(self.table_top_sdf)

    self.create_scene()
    self.init_joints()

    ## Weld everything together and visualize! 
    # self.plant.WeldFrames(self.plant.world_frame(), self.plant.GetFrameByName("base"),
    #                 RigidTransform(RollPitchYaw(0,0,0), [0.5, 0.2, 0.0]))

    ## Left Arm + Hand
    self.plant.WeldFrames(self.plant.world_frame(), self.plant.GetFrameByName("left_panda_link0"),
                    RigidTransform(RollPitchYaw(0,0,0), [0.06975, -0.27, 0.5]))

    self.plant.WeldFrames(self.plant.GetFrameByName("left_panda_link8"), self.plant.GetFrameByName("left_gripper_base"),
                    RigidTransform(RollPitchYaw(0, 3.141592653589793, -0.7853981633974483), [0.0, 0.0, 0.0]))
    ## Right Arm + Hand
    self.plant.WeldFrames(self.plant.world_frame(), self.plant.GetFrameByName("right_panda_link0"),
                    RigidTransform(RollPitchYaw(0,0,0), [0.06975, 0.27, 0.5]))

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
  def register_rbgd_sensor(self):
    """ Registers a RGBD sensor to the plant
    """
    pass
  
  def add_manipuland_from_file(self):
    """ Adds a single object to the plant from a file
    """
  

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
      self.set_default_angle_for_instance(self.left_panda_arm_instance, self.arm_state['home'])
      self.set_default_angle_for_instance(self.right_panda_arm_instance, self.arm_state['home'])
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

  def register_hand_controllers(self):
    # if self.hand_controller_type == "pid":
    #   hand_controller = self.hand_pid_controller()
    #   self.builder.Connect(hand_controller.get_output_port_control(), self.plant.get_actuation_input_port(self.hand_instance))
    #   passthrough_state = self.builder.AddSystem(PassThrough([0] * self.hand_actuator_dim))
    #   self.builder.Connect(self.plant.get_state_output_port(self.hand_instance), hand_controller.get_input_port_estimated_state())
    #   desired_state_from_position_hand = self.builder.AddSystem(
    #       StateInterpolatorWithDiscreteDerivative(self.hand_actuator_dim, self.timestep, suppress_initial_transient=True)
    #   )
    #   self.builder.Connect(passthrough_state.get_output_port(), desired_state_from_position_hand.get_input_port())
    #   self.builder.Connect(
    #       desired_state_from_position_hand.get_output_port(), hand_controller.get_input_port_desired_state()
    #   )
    #   self.builder.ExportInput(passthrough_state.get_input_port(), "q_robot_commanded")

    # elif self.hand_controller_type == "impedance":
    right_hand_controller = self.builder.AddSystem(HandControllerAvatar(
    plant=self.plant,
    model_instance=self.right_gripper_instance,
    damping_ratio=1,
    hand = "right",
    controller_mode="impedance"))           
    
    self.builder.Connect(right_hand_controller.get_output_port_control(), self.plant.get_actuation_input_port(self.right_gripper_instance))
    # passthrough_state = self.builder.AddSystem(PassThrough([0] * self.hand_joint_dim))
    self.builder.Connect(self.plant.get_state_output_port(self.right_gripper_instance), right_hand_controller.get_input_port_estimated_state())

    right_control_passthrough = self.builder.AddSystem(PassThrough([0] * self.hand_actuator_dim))
    self.builder.Connect(right_control_passthrough.get_output_port(), right_hand_controller.get_input_port_desired_state())
    self.builder.ExportInput(right_control_passthrough.get_input_port(), "right_q_robot_commanded")

    torque_passthrough = self.builder.AddSystem(PassThrough([0] * self.hand_joint_dim))
    self.builder.Connect(torque_passthrough.get_output_port(), right_hand_controller.get_torque_feedforward_port_control())
    self.builder.ExportInput(torque_passthrough.get_input_port(), "right_hand_feedforward_torque")
    self.builder.ExportOutput(right_hand_controller.get_output_port_control(), "right_hand_controller_output")
    self.builder.ExportOutput(self.plant.get_state_output_port(self.right_gripper_instance), "right_hand_state")
    
    left_hand_controller = self.builder.AddSystem(HandControllerAvatar(
    plant=self.plant,
    model_instance=self.left_gripper_instance,
    damping_ratio=1,
    hand = "left",
    controller_mode="impedance"))
    self.builder.Connect(left_hand_controller.get_output_port_control(), self.plant.get_actuation_input_port(self.left_gripper_instance))
    self.builder.Connect(self.plant.get_state_output_port(self.left_gripper_instance), left_hand_controller.get_input_port_estimated_state())

    left_control_passthrough = self.builder.AddSystem(PassThrough([0] * self.hand_actuator_dim))
    self.builder.Connect(left_control_passthrough.get_output_port(), left_hand_controller.get_input_port_desired_state())
    self.builder.ExportInput(left_control_passthrough.get_input_port(), "left_q_robot_commanded")

    torque_passthrough = self.builder.AddSystem(PassThrough([0] * self.hand_joint_dim))
    self.builder.Connect(torque_passthrough.get_output_port(), left_hand_controller.get_torque_feedforward_port_control())
    self.builder.ExportInput(torque_passthrough.get_input_port(), "left_hand_feedforward_torque")
    self.builder.ExportOutput(left_hand_controller.get_output_port_control(), "left_hand_controller_output")
    self.builder.ExportOutput(self.plant.get_state_output_port(self.left_gripper_instance), "left_hand_state")

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

  def debug_print(self, context, i):
    try:
      # print(f"timestep: {i}")
      # print("Panda state q: ", self.diagram.GetOutputPort("panda_state").Eval(context))
      # print("Desired state q_d", self.diagram.GetOutputPort("panda_desired_state").Eval(context))
      # print("Controller output:", self.diagram.GetOutputPort("controller_output").Eval(context))
      # print("Hand state q: ", self.diagram.GetOutputPort("hand_state").Eval(context))
      # print("Hand controller output:", self.diagram.GetOutputPort("hand_controller_output").Eval(context))
      # print(f"Graivity force: {self.plant.CalcGravityGeneralizedForces(self.plant.GetMyContextFromRoot(self.context))}")
      
      # print("===========================================")
      pass
    except RuntimeError as e:
      print(e)
      pass

  # def finger_state_callback(self, msg):
  #   self.desired_finger_pos = -1.0 * np.array([msg.position[1], msg.position[2], msg.position[0]])

  # def eff_pose_callback(self, msg):
  #   self.desired_eff_pose = np.array([msg.position.x, msg.position.y, msg.position.z, msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

  # def ros_data_receiver(self):
  #   rospy.init_node('ros_data_receiver', anonymous=True)
  #   rospy.Subscriber("/right_glove_joint_states", JointState, self.finger_state_callback)
  #   rospy.Subscriber("/right_arm_pose", Pose, self.eff_pose_callback)
  
  def simulate(self):
    simulator = Simulator(self.diagram, self.context)

    context = simulator.get_context()
    left_hand_state_input_port = self.diagram.GetInputPort("left_q_robot_commanded")
    right_hand_state_input_port = self.diagram.GetInputPort("right_q_robot_commanded")


    finger_states = [0.0, 0.0, 0.0]
    if self.motion == "stable":
        if self.hand_state == "open":
            finger_states = [0.0, 0.0, 0.0]
        else:
            finger_states = [-0.7, -1.2, -1.0]

        left_hand_state_input_port.FixValue(self.context, finger_states)
        print("Left Hand Input: ", left_hand_state_input_port.Eval(context))
        right_hand_state_input_port.FixValue(self.context, finger_states)
        print("Right Hand Input: ", right_hand_state_input_port.Eval(context))
        if self.arm_controller_type == "inv_dyn":
            target_joints = self.arm_state["home"]
            arm_state_input_port = self.diagram.GetInputPort("panda_joint_commanded")
            arm_state_input_port.FixValue(self.context, target_joints)

        elif self.arm_controller_type == "pose":
            left_desired_eff_pose_port = self.diagram.GetInputPort("left_desired_eff_pose")
            right_desired_eff_pose_port = self.diagram.GetInputPort("right_desired_eff_pose")

            left_desired_eff_vel_port  = self.diagram.GetInputPort("left_desired_eff_velocity")
            right_desired_eff_vel_port  = self.diagram.GetInputPort("right_desired_eff_velocity")

            left_desired_eff_pose_port.FixValue(
                context,
                RigidTransform(
                    RollPitchYaw(-1.2080553034387211, -0.3528716676941929, -0.847106272058664),
                    [1.460e-01, 0, 7.061e-01],
                ),
            )
            
            right_desired_eff_pose_port.FixValue(
                context,
                RigidTransform(
                    RollPitchYaw(-1.2080553034387211, -0.3528716676941929, -0.847106272058664),
                    [1.460e-01, 0, 7.061e-01],
                ),
            )
            left_desired_eff_vel_port.FixValue(context, SpatialVelocity(np.zeros(6)))
            right_desired_eff_vel_port.FixValue(context, SpatialVelocity(np.zeros(6)))

        elif self.arm_controller_type == "impedance":
            desired_joint_position_port = self.diagram.GetInputPort("panda_joint_commanded")
            target_joints = self.arm_state["q1"]
            # target_joints[0] -= 1
            # target_joints[1] -= 1.1
            desired_joint_position_port.FixValue(context, target_joints)

        # check stableness
        # simulator.AdvanceTo(10000)
        simulator.set_target_realtime_rate(1.0)

        # for jnt_idx in self.plant.GetJointActuatorIndices(self.hand_instance):
        #     jnt_act = self.plant.get_joint_actuator(jnt_idx)
        #     print("joint actuator name: ", jnt_act.name())
        
        i = 0
        print("Press Escape to stop the simulation")
        self.meshcat.AddButton("Stop Simulation", "Escape")
        self.meshcat.AddButton("Next Step", "Enter")
        simulator.AdvanceTo(0.0001)
        self.debug_print(context, i)

        t = 0
        dt = 0.001
        while self.meshcat.GetButtonClicks("Stop Simulation") < 1:
            # if n_cmd == self.meshcat.GetButtonClicks("Next Step"):
                # continue
            # simulator.AdvanceTo(self.meshcat.GetButtonClicks("Next Step")*dt)
            # self.debug_print(context)
            # n_cmd = self.meshcat.GetButtonClicks("Next Step")
            t = t + dt
            simulator.AdvanceTo(t)
            i += 1
            # print("finger command: ", finger_states)
            self.debug_print(context, i)
            # print(hand_act_p.HasValue(context))
                    
        self.meshcat.DeleteButton("Stop Simulation")

    elif self.motion == 'arm_teleop':

        simulator.set_target_realtime_rate(1)
        t = 0
        dt = 0.1
        desired_eff_pose_port = self.diagram.GetInputPort("desired_eff_pose")
        desired_eff_vel_port = self.diagram.GetInputPort("desired_eff_velocity")
        desired_eff_vel_port.FixValue(context, SpatialVelocity(np.zeros(6)))
        endeffector_rot = np.array([1.57, 1.57, 0.0])
        endeffector_trans = np.array([0.4, -0.1, 0.3])
        joint_pos_lower_limit = np.array(self.hand_joint_lower_limit[7:10])
        joint_pos_upper_limit = np.array(self.hand_joint_upper_limit[7:10])
        finger_states = np.array([0.0, 0.0, 0.0])

        self.desired_eff_pose[6] = 1.0
        rot_init = scipy.spatial.transform.Rotation.from_euler('xyz', [1.57, 1.57, 0.0])

        while not rospy.is_shutdown():
            endeffector_trans = self.desired_eff_pose[:3] + np.array([0.4, -0.1, 0.3])

            rot_relative = scipy.spatial.transform.Rotation.from_quat(self.desired_eff_pose[3:])
            rot = rot_relative * rot_init
            rot_euler = rot.as_euler('xyz')
            endeffector_rot = rot_euler

            desired_eff_pose_port.FixValue(
                context,
                RigidTransform(RollPitchYaw(*endeffector_rot), endeffector_trans),
            )

            finger_states = self.desired_finger_pos

            np.clip(finger_states, joint_pos_lower_limit, joint_pos_upper_limit, out=finger_states)
            hand_state_input_port.FixValue(context, finger_states)


            # step simulator
            t = t + dt
            simulator.AdvanceTo(t)

    else:
        # TODO(lirui): create another script
        from teleop_utils import get_keyboard_input, get_vr_input

        print("Press Escape to stop the simulation")
        self.meshcat.AddButton("Stop Simulation", "Escape")
        self.meshcat.AddButton("Next Step", "Enter")
        print("use keyboard teleop: use w,s,a,d,q,e for arm and [,],;,',.,/ for fingers")

        simulator.set_target_realtime_rate(1)
        t = 0
        dt = 0.001
        desired_eff_pose_port = self.diagram.GetInputPort("desired_eff_pose")
        desired_eff_vel_port = self.diagram.GetInputPort("desired_eff_velocity")
        desired_eff_vel_port.FixValue(context, SpatialVelocity(np.zeros(6)))
        endeffector_rot = np.array([1.57, 1.57, 0.0])
        endeffector_trans = np.array([0.4, -0.1, 0.3])
        joint_pos_lower_limit = np.array(self.hand_joint_lower_limit[7:10])
        joint_pos_upper_limit = np.array(self.hand_joint_upper_limit[7:10])
        finger_states = np.array([0.0, 0.0, 0.0])

        if self.teleop_type == "vr":
            # initialize the goal gripper
            vis_hand_pose(self.meshcat, np.eye(4), "hand_goal_pose", load=True)

        while self.meshcat.GetButtonClicks("Stop Simulation") < 1:
            if self.teleop_type == "keyboard":
                action, finger = get_keyboard_input()
                endeffector_trans += action[:3]
                endeffector_rot += action[3:]
            else:
                action, finger = get_vr_input()
                endeffector_trans[:] = action[:3]
                endeffector_rot[:] = action[3:]
                vis_hand_pose(self.meshcat, unpack_action(action), "hand_goal_pose")

            desired_eff_pose_port.FixValue(
                context,
                RigidTransform(RollPitchYaw(*endeffector_rot), endeffector_trans),
            )

            # could switch to individual finger as well
            if finger == "thumb_up":
                finger_states += 0.00, 0.1, 0.00
            elif finger == "thumb_down":
                finger_states -= 0.00, 0.1, 0.00
            elif finger == "thumb_open":
                finger_states += 0.1, 0.00, 0.00
            elif finger == "thumb_close":
                finger_states -= 0.1, 0.00, 0.00
            elif finger == "index_open":
                finger_states += 0.00, 0.00, 0.1
            elif finger == "index_close":
                finger_states -= 0.00, 0.00, 0.1

            np.clip(finger_states, joint_pos_lower_limit, joint_pos_upper_limit, out=finger_states)
            hand_state_input_port.FixValue(context, finger_states)


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


@click.command()
@click.option('--timestep', default=0.001, help='Timestep for simulation')
@click.option('--hand_controller_type', default='impedance', help='Type of hand controller: impedance or pid')
@click.option('--hand_state', default='open', help='State of hand: open or closed')
@click.option('--arm_controller_type', default='pose', help='Type of arm controller: inv_dym, pose, impedance')
@click.option('--motion', default='stable', help='Type of motion: stable or teleop')
@click.option('--teleop_type', default='keyboard', help='Type of teleop: keyboard or vr')
@click.option('--debug', default=False, help='Debug mode')
@click.option('--plot', default=False, help='Plot mode')
def main(timestep, hand_controller_type, hand_state, arm_controller_type, motion, teleop_type, debug, plot):
  click.echo('Timestep: %s' % timestep)
  click.echo('Hand controller type: %s' % hand_controller_type)
  click.echo('Hand state: %s' % hand_state)
  click.echo('Arm controller type: %s' % arm_controller_type)
  click.echo('Motion: %s' % motion)
  click.echo('Teleop type: %s' % teleop_type)
  click.echo('Debug: %s' % debug)
  click.echo('Plot: %s' % plot)

  avatar_drake_env = AvatarDrakeEnv(timestep, hand_controller_type, hand_state, arm_controller_type, motion, teleop_type, debug, plot)
  avatar_drake_env.simulate()
  avatar_drake_env.close()

if __name__ == "__main__":
  main()