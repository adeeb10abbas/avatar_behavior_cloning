import numpy as np
from pydrake.geometry import (
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
    StartMeshcat,
)

from avatar_behavior_cloning.controllers.impedance_controller_avatar_hand import *
# from controllers.impedance_controller_avatar_hand import HandControllerAvatar
# from pydrake.systems.controllers import JointStiffnessController, InverseDynamics
# from utils.geometry_utils import *
# from utils.drake_utils import *
# from pydrake.math import RigidTransform, RollPitchYaw
# from pydrake.multibody.parsing import Parser
# from pydrake.systems.analysis import Simulator
# from pydrake.systems.framework import DiagramBuilder, OutputPort
# from pydrake.multibody.plant import AddMultibodyPlant, MultibodyPlantConfig, MultibodyPlant
# from pydrake.multibody.tree import SpatialInertia, ModelInstanceIndex

# import pydot
# import click
# import time

# import rospy
# from sensor_msgs.msg import JointState
# from geometry_msgs.msg import Pose
# from scipy.spatial.transform import Rotations

# class ModelInformation:
#   def __init__(self):
#     self.model_path = ""
#     self.model_instance = 0
#     self.parent_frame = ""
#     self.child_frame = ""
#     self.transform = RigidTransform()

  

# class AvatarDrakeEnv:
#   def __init__(self, timestep, hand_controller_type, hand_state, arm_controller_type, motion, teleop_type, debug, plot):
    
#     ############################################# 
#     # Set up parameters
#     self.timestep = timestep
#     self.motion = motion
#     self.teleop_type = teleop_type
#     self.debug = debug
#     self.plot = plot

#     self.arm_controller_type = arm_controller_type
#     self.arm_state = {"zero": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
#                       "home": [0.1, 0.1, 0.0, -2.7, 1.5708, 1.5708, -2.0],
#                       "q1":   [0.1, 0.1, -0.5, -2.356, 0.0, 0, 0.785],
#                       "q2":   [1.0, 0.5, -1.1, -2.356, 0.0, 0, 0.785]}
#     self.panda_joints = 7
#     self.panda_kp = [5000] * self.panda_joints
#     self.panda_ki = [1] *  self.panda_joints
#     self.panda_kd = [100] * self.panda_joints

#     # Change this to 3 if using mimic
#     self.hand_controller_type = hand_controller_type
#     self.hand_state = hand_state
#     self.hand_joint_dim = 10
#     self.hand_actuator_dim = 3
#     self.hand_kp = []
#     self.hand_kd = []
#     self.panda_model_urdf = "models/panda_arm.urdf"
#     self.hand_model_urdf  = "models/avatar_gripper_right_nomimic.urdf"
#     self.table_top_sdf = "models/env_objects/table_top.sdf"


#     #############################################
#     # Initialize
#     self.meshcat = StartMeshcat()
#     self.builder = DiagramBuilder()
#     self.multibody_plant_config = MultibodyPlantConfig(time_step=self.timestep, discrete_contact_solver="sap")
#     self.plant, self.scene_graph = AddMultibodyPlant(config=self.multibody_plant_config, builder=self.builder)
#     self.parser = Parser(self.plant)
#     self.parser.package_map().Add("avatar_gripper_description", "../avatar_simulation/avatar_gripper_description")
#     # separate plant for controller
#     self.controller_plant = MultibodyPlant(time_step=self.timestep)
#     self.controller_parser = Parser(self.controller_plant)
#     self.env_plant = MultibodyPlant(time_step=self.timestep)
#     self.env_parser = Parser(self.env_plant)
#     self.hand_controller_plant = MultibodyPlant(time_step=self.timestep)
#     self.hand_controller_parser = Parser(self.hand_controller_plant)
#     self.hand_joint_lower_limit = []
#     self.hand_joint_upper_limit = []
    
#     self.register_whole_plant()
#     self.register_panda_controller_plant()
#     self.register_hand_controller_plant()
#     # self.create_scene()

#     #############################################
#     # Add arm and hand controllers and connect diagram
#     self.register_arm_controllers()
#     self.register_hand_controllers()

#     #############################################
#     # Arm and gripper states
#     self.desired_eff_pose = np.zeros(7)
#     self.desired_finger_pos = np.zeros(3)
  
#     #############################################
#     visualizer = MeshcatVisualizer.AddToBuilder(
#         self.builder,
#         self.scene_graph,
#         self.meshcat,  # kIllustration, kProximity
#         MeshcatVisualizerParams(role=Role.kIllustration, delete_on_initialization_event=False),
#     )

#     # Add contact visualizer
#     cparams = ContactVisualizerParams()
#     cparams.force_threshold = 1e-2
#     cparams.newtons_per_meter = 1e6  # .0 # 1.0
#     cparams.newton_meters_per_meter = 1e1
#     cparams.radius = 0.005  # 0.01
#     contact_visualizer = ContactVisualizer.AddToBuilder(self.builder, self.plant, self.meshcat, cparams)

#     if self.plot:
#         self.plotter = start_logger()

#     # Build the diagram
#     self.diagram = self.builder.Build()
#     self.context = self.diagram.CreateDefaultContext()
#     pydot.graph_from_dot_data(self.diagram.GetGraphvizString())[0].write_svg("diagram.svg")

#     self.panda_state = self.diagram.GetOutputPort("panda_state")

#   def create_scene(self):
#     self.table_instance = self.parser.AddModelFromFile(self.table_top_sdf)
#     self.plant.WeldFrames(
#       self.plant.world_frame(),
#       self.plant.GetFrameByName("table_top_center"),
#       RigidTransform(RollPitchYaw(0,0,0), [0.5, 0.2, 0.5])
#     )
#     container_instance = self.parser.AddModelFromFile("models/env_objects/container.sdf")
#     self.plant.WeldFrames(
#       self.plant.GetFrameByName("table_top_center"),
#       self.plant.GetFrameByName("container_center"),
#       RigidTransform(RollPitchYaw(0,0,0), [0.0, 0.3, 0.01])
#     )
#     cylinder_instance = self.parser.AddModelFromFile("models/env_objects/cylinder.sdf")
#     cube_instance = self.parser.AddModels("models/env_objects/cube.sdf")

#   def setup_manipulation_scene(self):
#     """ Adds panda, hand, table and object to the scene
#     """
#     pass

#   def setup_default_state(self):
#     """ Sets up the default state for the chosen setup
#     """
#     pass
  
#   def setup_random_state(self):
#     """ Sets up a random state for the chosen setup
#     """
#     pass
  
#   def register_whole_plant(self):
#     """ Registers the whole plant for visualization
#     """
#     # Add model for visualization
#     self.panda_instance = self.parser.AddModelFromFile(self.panda_model_urdf)
#     self.hand_instance = self.parser.AddModelFromFile(self.hand_model_urdf)

#     self.create_scene()
#     self.init_joints()

#     self.plant.WeldFrames(
#       self.plant.world_frame(), 
#       self.plant.GetFrameByName("panda_link0"),
#       RigidTransform(RollPitchYaw([0.0, 0.0, 0.0]),[0.0, 0.0, 0.5])
#     )
    
#     self.plant.WeldFrames(
#         self.plant.GetFrameByName("panda_link8"),
#         self.plant.GetFrameByName("right_gripper_base"),
#         RigidTransform(RollPitchYaw(np.pi, 0, 3*np.pi/4), [0.0, 0, 0.0]),
#     )
#     self.plant.Finalize()

#     plant_context = self.plant.CreateDefaultContext()
#     X_WorldTable = self.plant.GetFrameByName("table_top_center").CalcPoseInWorld(plant_context)
#     X_TableCylinder = RigidTransform(
#         RollPitchYaw(np.asarray([0, 0, 0]) * np.pi / 180), p=[0,-0.2,0.1])
#     X_WorldCylinder = X_WorldTable.multiply(X_TableCylinder)
#     self.plant.SetDefaultFreeBodyPose(self.plant.GetBodyByName("cylinder_link"), X_WorldCylinder)
#     print("plant q size: ", self.plant.num_positions())

#   def register_panda_controller_plant(self):
#     """ Registers a plant specifically for the panda controller with attached rigib body that has the same inertia as 
#         the hand
#     """
#     # Add model for control
#     self.panda_controller_model = self.controller_parser.AddModelFromFile(self.panda_model_urdf)
#     self.controller_plant.WeldFrames(
#       self.controller_plant.world_frame(), 
#       self.controller_plant.GetFrameByName("panda_link0"))
    
#     # TODO(rui): change this to real hand inertia stats
#     hand_spatial_inertial = SpatialInertia.SolidEllipsoidWithMass(2, 0.05, 0.15, 0.07)
#     hand_equivalent_body  = self.controller_plant.AddRigidBody(
#       "hand_equivalent_body", 
#       self.panda_controller_model,
#       hand_spatial_inertial)
    
#     self.controller_plant.WeldFrames(
#       self.controller_plant.GetFrameByName("panda_link8"),
#       hand_equivalent_body.body_frame(), 
#       RigidTransform(RollPitchYaw(np.pi, 0, 3*np.pi/4), [0.0, 0, 0.0]),
#     )
#     self.controller_plant.set_name('panda_controller_plant')
#     self.controller_plant.Finalize()
#     # panda's q is 7
#     print("panda_controller q size: ", self.controller_plant.num_positions())
#     # if self.debug:
#     #     simplify_plant(self.plant, self.scene_graph)
    
    

#   def register_hand_controller_plant(self):
#     """ Registers a plant specifically for the hand controller without the Panda arm
#     """
#     self.hand_controller_model = self.hand_controller_parser.AddModelFromFile(self.hand_model_urdf)
#     self.hand_controller_plant.WeldFrames(
#         self.hand_controller_plant.world_frame(),
#         self.hand_controller_plant.GetFrameByName("right_gripper_base"),
#         RigidTransform(RollPitchYaw(np.pi, 0, 3*np.pi/4), [0.0, 0, 0.0]),
#     )
#     self.hand_controller_plant.set_name('hand_controller_plant')
#     self.hand_controller_plant.Finalize()
#     print("hand_controller q size: ", self.hand_controller_plant.num_positions())
  
#   def register_rbgd_sensor(self):
#     """ Registers a RGBD sensor to the plant
#     """
#     pass
  
#   def add_manipuland_from_file(self):
#     """ Adds a single object to the plant from a file
#     """
  
  
#   def init_joints(self):
#     # Set default joints
#     # q0 = self.arm_state['home']
#     index = 0
#     for joint_index in self.plant.GetJointIndices(self.panda_instance):
#         joint = self.plant.get_mutable_joint(joint_index)
#         if isinstance(joint, RevoluteJoint):
#             joint.set_default_angle(self.arm_state["home"][index])
#             # print(joint.position_lower_limit(), joint.position_upper_limit())
#             # print(joint.velocity_lower_limit(), joint.velocity_upper_limit())
#             # print(joint.acceleration_lower_limit(), joint.acceleration_upper_limit())
#             # print(joint.set_default_damping(5))
#             index += 1

#     q0 = [0] * self.hand_joint_dim
#     index = 0
#     for joint_index in self.plant.GetJointIndices(self.hand_instance):
#         joint = self.plant.get_mutable_joint(joint_index)
#         print('joint name: ', joint.name())
#         if isinstance(joint, RevoluteJoint):
#             joint.set_default_angle(q0[index])
#             print('joint position lower limit: ', joint.position_lower_limit())
#             print('joint position upper limit: ', joint.position_upper_limit())
#             self.hand_joint_lower_limit.append(joint.position_lower_limit())
#             self.hand_joint_upper_limit.append(joint.position_upper_limit())
#             index += 1
#     # self.plant.Finalize()
#     # context = self.plant.CreateDefaultContext()
#     # print("Test: ", self.plant.get_state_output_port(self.hand_instance).GetFullDescription())

#   def register_arm_controllers(self):
#     if self.arm_controller_type == "inv_dyn":
#       # This would fail as the plant contains more than the arm
#       panda_controller = self.builder.AddSystem(
#           InverseDynamicsController(
#             self.controller_plant, 
#             self.panda_kp, 
#             self.panda_ki, 
#             self.panda_kd, 
#             False)
#       )
#       panda_controller.set_name("panda_inverse_dynamics_controller")
#       self.builder.Connect(
#           self.plant.get_state_output_port(self.panda_instance), 
#           panda_controller.get_input_port_estimated_state()
#       )

#       panda_positions = self.builder.AddSystem(PassThrough([0] * self.panda_joints))
#       desired_state_from_position = self.builder.AddSystem(
#           StateInterpolatorWithDiscreteDerivative(self.panda_joints, self.timestep, suppress_initial_transient=True)
#       )
#       self.builder.Connect(
#           desired_state_from_position.get_output_port(),
#           panda_controller.get_input_port_desired_state(),
#       )
      
#       self.builder.Connect(
#           panda_positions.get_output_port(),
#           desired_state_from_position.get_input_port(),
#       )
      
      
#       self.builder.Connect(
#           panda_controller.get_output_port_control(),
#           self.plant.get_actuation_input_port(self.panda_instance)
#       )
      
#       # Declare the input ports of panda_positions to be the input ports of the entire diagram, named as "panda_joint_commanded"
#       self.builder.ExportInput(panda_positions.get_input_port(), "panda_joint_commanded")
#       self.builder.ExportOutput(panda_controller.get_output_port_control(), "controller_output")
#       self.builder.ExportOutput(
#           desired_state_from_position.get_output_port(), "panda_desired_state"
#       )

#     elif self.arm_controller_type == "pose":
#       panda_controller = self.builder.AddSystem(
#           PoseControllerAvatar(
#             self.plant, 
#             self.panda_instance, 
#             self.plant.GetFrameByName("panda_link0"), 
#             self.plant.GetFrameByName("right_gripper_base"))
#       )
#       desired_eff_pose = self.builder.AddSystem(PassThrough(AbstractValue.Make(RigidTransform())))
#       desired_eff_velocity = self.builder.AddSystem(PassThrough(AbstractValue.Make(SpatialVelocity())))
#       self.builder.Connect(
#           self.plant.get_state_output_port(self.panda_instance),
#           panda_controller.get_plant_state_input_port(),
#       )

#       self.builder.Connect(
#           desired_eff_pose.get_output_port(),
#           panda_controller.get_state_desired_port(),
#       )
#       self.builder.Connect(
#           desired_eff_velocity.get_output_port(),
#           panda_controller.get_velocity_desired_port(),
#       )

#       self.builder.ExportInput(desired_eff_pose.get_input_port(), "desired_eff_pose")
#       self.builder.ExportInput(desired_eff_velocity.get_input_port(), "desired_eff_velocity")

#       adder = self.builder.AddSystem(Adder(2, panda_controller.nu))
#       torque_passthrough = self.builder.AddSystem(PassThrough([0] * panda_controller.nu))
#       self.builder.Connect(panda_controller.get_output_port(), adder.get_input_port(0))
#       self.builder.Connect(torque_passthrough.get_output_port(), adder.get_input_port(1))
#       self.builder.ExportInput(torque_passthrough.get_input_port(), "feedforward_torque")
#       self.builder.Connect(adder.get_output_port(), self.plant.get_actuation_input_port(self.panda_instance))      
    
#     elif self.arm_controller_type == "impedance":
#       # TODO(@Rui): not working properly now, things get exploded during simulation
#       panda_controller = self.builder.AddSystem(JointStiffnessController(
#         self.controller_plant,
#         [400, 400, 500, 500, 250, 100, 100],
#         # [1.4, 4, 1.4, 2, 0.5, 0.1, 0.1]
#         [30, 10, 50, 50, 10, 1, 1]
#       ))

#       panda_controller.set_name("panda_joint_stiffness_controller")
#       self.builder.Connect(self.plant.get_state_output_port(self.panda_instance),
#                             panda_controller.get_input_port_estimated_state())
      
#       # Add in feedforward torque
#       # adder = self.builder.AddSystem(Adder(2, self.panda_joints))
#       # self.builder.Connect(panda_controller.get_output_port_control(), adder.get_input_port(0))
      
#       position_command = self.builder.AddSystem(PassThrough([0] * self.panda_joints))
#       self.builder.ExportInput(position_command.get_input_port(), "panda_joint_commanded")

#       # self.builder.Connect(position_passthrough.get_output_port(), 
#       #                      adder.get_input_port(1))
#       self.builder.Connect(
#           panda_controller.get_output_port_generalized_force(),
#           self.plant.get_actuation_input_port(self.panda_instance)
#       )
#       desired_state_from_position = self.builder.AddSystem(
#           StateInterpolatorWithDiscreteDerivative(self.panda_joints, self.timestep, suppress_initial_transient=True)
#       )
#       self.builder.Connect(
#           desired_state_from_position.get_output_port(),
#           panda_controller.get_input_port_desired_state(),
#       )
#       print("Desired state from position output size: ", desired_state_from_position.get_output_port().size())
#       print("Panda controller input size: ", panda_controller.get_input_port_desired_state().size())
#       self.builder.Connect(
#           position_command.get_output_port(),
#           desired_state_from_position.get_input_port(),
#       )
#       self.builder.ExportOutput(
#           panda_controller.get_output_port_generalized_force(), "controller_output"
#       )
#       self.builder.ExportOutput(
#           desired_state_from_position.get_output_port(), "panda_desired_state"
#       )
    
#     elif self.arm_controller_type == "pure_gravity_compensation":
#       panda_controller = self.builder.AddSystem(InverseDynamics(
#         self.controller_plant,
#         InverseDynamics.kGravityCompensation
#       ))
#       panda_controller.set_name("panda_gravity_compensation_controller")
#       self.builder.Connect(self.plant.get_state_output_port(self.panda_instance),
#                             panda_controller.get_input_port_estimated_state())
#       self.builder.ExportOutput(panda_controller.get_output_port_force(), "controller_output")
#       self.builder.Connect(panda_controller.get_output_port_force(), 
#                            self.plant.get_actuation_input_port(self.panda_instance))
      
      
    
#     self.builder.ExportOutput(self.plant.get_state_output_port(self.panda_instance), "panda_state")
    

#   def register_hand_controllers(self):
#     if self.hand_controller_type == "pid":
#       hand_controller = self.hand_pid_controller()
#       self.builder.Connect(hand_controller.get_output_port_control(), self.plant.get_actuation_input_port(self.hand_instance))
#       passthrough_state = self.builder.AddSystem(PassThrough([0] * self.hand_actuator_dim))
#       self.builder.Connect(self.plant.get_state_output_port(self.hand_instance), hand_controller.get_input_port_estimated_state())
#       desired_state_from_position_hand = self.builder.AddSystem(
#           StateInterpolatorWithDiscreteDerivative(self.hand_actuator_dim, self.timestep, suppress_initial_transient=True)
#       )
#       self.builder.Connect(passthrough_state.get_output_port(), desired_state_from_position_hand.get_input_port())
#       self.builder.Connect(
#           desired_state_from_position_hand.get_output_port(), hand_controller.get_input_port_desired_state()
#       )
#       self.builder.ExportInput(passthrough_state.get_input_port(), "q_robot_commanded")

#     elif self.hand_controller_type == "impedance":
      
#       # hand_controller = self.builder.AddSystem(HandControllerAvatar(
#       #   plant=self.hand_controller_plant, 
#       #   model_instance=self.hand_controller_plant.GetModelInstanceByName("avatar_gripper_3f_model"), 
#       #   damping_ratio=1, 
#       #   controller_mode="inverse_dynamics"))
#       hand_controller = self.builder.AddSystem(HandControllerAvatar(
#         plant=self.plant,
#         model_instance=self.hand_instance,
#         damping_ratio=1,
#         controller_mode="impedance"))
                                               
      
#       self.builder.Connect(hand_controller.get_output_port_control(), self.plant.get_actuation_input_port(self.hand_instance))
#       passthrough_state = self.builder.AddSystem(PassThrough([0] * self.hand_joint_dim))
#       self.builder.Connect(self.plant.get_state_output_port(self.hand_instance), hand_controller.get_input_port_estimated_state())

#       control_passthrough = self.builder.AddSystem(PassThrough([0] * self.hand_actuator_dim))
#       self.builder.Connect(control_passthrough.get_output_port(), hand_controller.get_input_port_desired_state())
#       self.builder.ExportInput(control_passthrough.get_input_port(), "q_robot_commanded")

#       torque_passthrough = self.builder.AddSystem(PassThrough([0] * self.hand_joint_dim))
#       self.builder.Connect(torque_passthrough.get_output_port(), hand_controller.get_torque_feedforward_port_control())
#       self.builder.ExportInput(torque_passthrough.get_input_port(), "hand_feedforward_torque")
#       self.builder.ExportOutput(hand_controller.get_output_port_control(), "hand_controller_output")
#       self.builder.ExportOutput(self.plant.get_state_output_port(self.hand_instance), "hand_state")
  
#   def hand_pid_controller(self):

#     stiffness = 3.0
#     state_projection_matrix = np.zeros((6, 20))
#     index = [0, 2, 1]

#     # joint / state:
#     # right_thumb_flex_motor_joint
#     # right_thumb_swivel_motor_joint
#     # right_index_flex_motor_joint
#     # right_thumb_knuckle_joint
#     # right_thumb_finger_joint
#     # right_thumb_swivel_joint
#     # right_index_knuckle_joint
#     # right_index_fingertip_joint
#     # right_middle_knuckle_joint
#     # right_middle_fingertip_joint

#     # actuator:
#     # right_thumb_flex_motor_joint
#     # right_index_flex_motor_joint
#     # right_thumb_swivel_motor_joint

#     state_projection_matrix[0, 0] = 1
#     state_projection_matrix[1, 2] = 1.0
#     state_projection_matrix[2, 1] = 1.0

#     # velocity
#     state_projection_matrix[3, 10] = 1
#     state_projection_matrix[4, 12] = 1.0
#     state_projection_matrix[5, 11] = 1.0
#     pid_controller = self.builder.AddSystem(
#         PidController(
#             state_projection=state_projection_matrix,
#             kp=np.ones(self.hand_actuator_dim) * stiffness,
#             ki=np.ones(self.hand_actuator_dim) * 0.01,
#             kd=np.ones(self.hand_actuator_dim) * 0.01,
#         )
#     )  # np.ones
#     return pid_controller

#   def debug_print(self, context, i):
#     try:
#       print(f"timestep: {i}")
#       # print("Panda state q: ", self.diagram.GetOutputPort("panda_state").Eval(context))
#       # print("Desired state q_d", self.diagram.GetOutputPort("panda_desired_state").Eval(context))
#       # print("Controller output:", self.diagram.GetOutputPort("controller_output").Eval(context))
#       # print("Hand state q: ", self.diagram.GetOutputPort("hand_state").Eval(context))
#       print("Hand controller output:", self.diagram.GetOutputPort("hand_controller_output").Eval(context))
#       # print(f"Graivity force: {self.plant.CalcGravityGeneralizedForces(self.plant.GetMyContextFromRoot(self.context))}")
      
#       print("===========================================")
#     except RuntimeError as e:
#       print(e)
#       pass

#   def finger_state_callback(self, msg):
#     self.desired_finger_pos = -1.0 * np.array([msg.position[1], msg.position[2], msg.position[0]])

#   def eff_pose_callback(self, msg):
#     self.desired_eff_pose = np.array([msg.position.x, msg.position.y, msg.position.z, msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

#   def ros_data_receiver(self):
#     rospy.init_node('ros_data_receiver', anonymous=True)
#     rospy.Subscriber("/right_glove_joint_states", JointState, self.finger_state_callback)
#     rospy.Subscriber("/right_arm_pose", Pose, self.eff_pose_callback)
  
#   def simulate(self):
#     simulator = Simulator(self.diagram, self.context)

#     context = simulator.get_context()
#     hand_state_input_port = self.diagram.GetInputPort("q_robot_commanded")


#     finger_states = [0.0, 0.0, 0.0]
#     if self.motion == "stable":
#         if self.hand_state == "open":
#             finger_states = [0.0, 0.0, 0.0]
#         else:
#             finger_states = [-0.7, -1.2, -1.0]

#         hand_state_input_port.FixValue(self.context, finger_states)
#         print("input: ", hand_state_input_port.Eval(context))
#         if self.arm_controller_type == "inv_dyn":
#             target_joints = self.arm_state["home"]
#             arm_state_input_port = self.diagram.GetInputPort("panda_joint_commanded")
#             arm_state_input_port.FixValue(self.context, target_joints)

#         elif self.arm_controller_type == "pose":
#             desired_eff_pose_port = self.diagram.GetInputPort("desired_eff_pose")
#             desired_eff_vel_port  = self.diagram.GetInputPort("desired_eff_velocity")
#             desired_eff_pose_port.FixValue(
#                 context,
#                 RigidTransform(
#                     RollPitchYaw(-1.2080553034387211, -0.3528716676941929, -0.847106272058664),
#                     [1.460e-01, 0, 7.061e-01],
#                 ),
#             )
#             desired_eff_vel_port.FixValue(context, SpatialVelocity(np.zeros(6)))
        
#         elif self.arm_controller_type == "impedance":
#             desired_joint_position_port = self.diagram.GetInputPort("panda_joint_commanded")
#             target_joints = self.arm_state["q1"]
#             # target_joints[0] -= 1
#             # target_joints[1] -= 1.1
#             desired_joint_position_port.FixValue(context, target_joints)

#         # check stableness
#         # simulator.AdvanceTo(10000)
#         simulator.set_target_realtime_rate(1.0)
#         for jnt_idx in self.plant.GetJointActuatorIndices(self.hand_instance):
#             jnt_act = self.plant.get_joint_actuator(jnt_idx)
#             print("joint actuator name: ", jnt_act.name())
        
#         i = 0
#         print("Press Escape to stop the simulation")
#         self.meshcat.AddButton("Stop Simulation", "Escape")
#         self.meshcat.AddButton("Next Step", "Enter")
#         simulator.AdvanceTo(0)
#         self.debug_print(context, i)

#         t = 0
#         dt = 0.001
#         while self.meshcat.GetButtonClicks("Stop Simulation") < 1:
#             # if n_cmd == self.meshcat.GetButtonClicks("Next Step"):
#                 # continue
#             # simulator.AdvanceTo(self.meshcat.GetButtonClicks("Next Step")*dt)
#             # self.debug_print(context)
#             # n_cmd = self.meshcat.GetButtonClicks("Next Step")
#             t = t + dt
#             simulator.AdvanceTo(t)
#             i += 1
#             print("finger command: ", finger_states)
#             self.debug_print(context, i)
#             # print(hand_act_p.HasValue(context))
                    
#         self.meshcat.DeleteButton("Stop Simulation")

#     elif self.motion == 'arm_teleop':

#         simulator.set_target_realtime_rate(1)
#         t = 0
#         dt = 0.1
#         desired_eff_pose_port = self.diagram.GetInputPort("desired_eff_pose")
#         desired_eff_vel_port = self.diagram.GetInputPort("desired_eff_velocity")
#         desired_eff_vel_port.FixValue(context, SpatialVelocity(np.zeros(6)))
#         endeffector_rot = np.array([1.57, 1.57, 0.0])
#         endeffector_trans = np.array([0.4, -0.1, 0.3])
#         joint_pos_lower_limit = np.array(self.hand_joint_lower_limit[7:10])
#         joint_pos_upper_limit = np.array(self.hand_joint_upper_limit[7:10])
#         finger_states = np.array([0.0, 0.0, 0.0])

#         self.desired_eff_pose[6] = 1.0
#         rot_init = Rotation.from_euler('xyz', [1.57, 1.57, 0.0])

#         while not rospy.is_shutdown():
#             endeffector_trans = self.desired_eff_pose[:3] + np.array([0.4, -0.1, 0.3])

#             rot_relative = Rotation.from_quat(self.desired_eff_pose[3:])
#             rot = rot_relative * rot_init
#             rot_euler = rot.as_euler('xyz')
#             endeffector_rot = rot_euler

#             desired_eff_pose_port.FixValue(
#                 context,
#                 RigidTransform(RollPitchYaw(*endeffector_rot), endeffector_trans),
#             )

#             finger_states = self.desired_finger_pos

#             np.clip(finger_states, joint_pos_lower_limit, joint_pos_upper_limit, out=finger_states)
#             hand_state_input_port.FixValue(context, finger_states)


#             # step simulator
#             t = t + dt
#             simulator.AdvanceTo(t)

#     else:
#         # TODO(lirui): create another script
#         from teleop_utils import get_keyboard_input, get_vr_input

#         print("Press Escape to stop the simulation")
#         self.meshcat.AddButton("Stop Simulation", "Escape")
#         self.meshcat.AddButton("Next Step", "Enter")
#         print("use keyboard teleop: use w,s,a,d,q,e for arm and [,],;,',.,/ for fingers")

#         simulator.set_target_realtime_rate(1)
#         t = 0
#         dt = 0.001
#         desired_eff_pose_port = self.diagram.GetInputPort("desired_eff_pose")
#         desired_eff_vel_port = self.diagram.GetInputPort("desired_eff_velocity")
#         desired_eff_vel_port.FixValue(context, SpatialVelocity(np.zeros(6)))
#         endeffector_rot = np.array([1.57, 1.57, 0.0])
#         endeffector_trans = np.array([0.4, -0.1, 0.3])
#         joint_pos_lower_limit = np.array(self.hand_joint_lower_limit[7:10])
#         joint_pos_upper_limit = np.array(self.hand_joint_upper_limit[7:10])
#         finger_states = np.array([0.0, 0.0, 0.0])

#         if self.teleop_type == "vr":
#             # initialize the goal gripper
#             vis_hand_pose(self.meshcat, np.eye(4), "hand_goal_pose", load=True)

#         while self.meshcat.GetButtonClicks("Stop Simulation") < 1:
#             if self.teleop_type == "keyboard":
#                 action, finger = get_keyboard_input()
#                 endeffector_trans += action[:3]
#                 endeffector_rot += action[3:]
#             else:
#                 action, finger = get_vr_input()
#                 endeffector_trans[:] = action[:3]
#                 endeffector_rot[:] = action[3:]
#                 vis_hand_pose(self.meshcat, unpack_action(action), "hand_goal_pose")

#             desired_eff_pose_port.FixValue(
#                 context,
#                 RigidTransform(RollPitchYaw(*endeffector_rot), endeffector_trans),
#             )

#             # could switch to individual finger as well
#             if finger == "thumb_up":
#                 finger_states += 0.00, 0.1, 0.00
#             elif finger == "thumb_down":
#                 finger_states -= 0.00, 0.1, 0.00
#             elif finger == "thumb_open":
#                 finger_states += 0.1, 0.00, 0.00
#             elif finger == "thumb_close":
#                 finger_states -= 0.1, 0.00, 0.00
#             elif finger == "index_open":
#                 finger_states += 0.00, 0.00, 0.1
#             elif finger == "index_close":
#                 finger_states -= 0.00, 0.00, 0.1

#             np.clip(finger_states, joint_pos_lower_limit, joint_pos_upper_limit, out=finger_states)
#             hand_state_input_port.FixValue(context, finger_states)


#             # step simulator
#             t = t + dt
#             simulator.AdvanceTo(t)

#             if self.plot:
#                 debug_joint = 1
#                 desired_joint = finger_states[debug_joint]
#                 plant_context = self.plant.GetMyMutableContextFromRoot(context)
#                 hand_joints = self.plant.GetPositions(plant_context, self.hand_instance)
#                 actual_joint  = hand_joints[debug_joint]
#                 update_line(self.plotter[0], self.plotter[1], [t, actual_joint], [t, desired_joint])


# @click.command()
# @click.option('--timestep', default=0.001, help='Timestep for simulation')
# @click.option('--hand_controller_type', default='impedance', help='Type of hand controller: impedance or pid')
# @click.option('--hand_state', default='open', help='State of hand: open or closed')
# @click.option('--arm_controller_type', default='pose', help='Type of arm controller: inv_dym, pose, impedance')
# @click.option('--motion', default='stable', help='Type of motion: stable or teleop')
# @click.option('--teleop_type', default='keyboard', help='Type of teleop: keyboard or vr')
# @click.option('--debug', default=False, help='Debug mode')
# @click.option('--plot', default=False, help='Plot mode')
# def main(timestep, hand_controller_type, hand_state, arm_controller_type, motion, teleop_type, debug, plot):
#   click.echo('Timestep: %s' % timestep)
#   click.echo('Hand controller type: %s' % hand_controller_type)
#   click.echo('Hand state: %s' % hand_state)
#   click.echo('Arm controller type: %s' % arm_controller_type)
#   click.echo('Motion: %s' % motion)
#   click.echo('Teleop type: %s' % teleop_type)
#   click.echo('Debug: %s' % debug)
#   click.echo('Plot: %s' % plot)

#   avatar_drake_env = AvatarDrakeEnv(timestep, hand_controller_type, hand_state, arm_controller_type, motion, teleop_type, debug, plot)
#   avatar_drake_env.ros_data_receiver()
#   avatar_drake_env.simulate()

# if __name__ == "__main__":
#   main()