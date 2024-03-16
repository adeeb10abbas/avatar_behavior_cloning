from pydrake.all import ModelVisualizer, PackageMap, StartMeshcat
import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import DiagramBuilder, MultibodyPlant, Parser, StartMeshcat, MultibodyPlantConfig, AddMultibodyPlant, Simulator
from pydrake.geometry import MeshcatVisualizer, MeshcatVisualizerParams, Role, StartMeshcat
from pydrake.math import RigidTransform, RollPitchYaw

from pydrake.math import RigidTransform, RollPitchYaw
from controller_ros_passthrough import PassthroughController

panda_arm = "package://drake/manipulation/models/franka_description/urdf/panda_arm.urdf"

### Initialization ### 
meshcat = StartMeshcat()
builder = DiagramBuilder()
multibody_plant_config = MultibodyPlantConfig(time_step=timestep, discrete_contact_solver="sap")
plant, scene_graph = AddMultibodyPlant(config=multibody_plant_config, builder=builder)
parser = Parser(plant)
###

left_controller_plant = MultibodyPlant(time_step=timestep)
right_controller_plant = MultibodyPlant(time_step=timestep)


left_parser = Parser(plant, "left")
left_hand_parser = Parser(plant, "left_hand")

right_parser = Parser(plant, "right")
right_hand_parser = Parser(plant, "right_hand")

# parser.SetAutoRenaming(True)
# left_parser.
left_parser.AddModelsFromUrl(panda_arm)
right_parser.AddModelsFromUrl(panda_arm)

left_hand_parser.AddModels("/home/adeebabbas/isolated/avatar_behavior_cloning/data/avatar/urdf/avatar_gripper_left_nomimic.urdf")
right_hand_parser.AddModels("/home/adeebabbas/isolated/avatar_behavior_cloning/data/avatar/urdf/avatar_gripper_right_nomimic.urdf")
# Add the robot to the scene
left_panda_arm_instance = plant.GetModelInstanceByName("left::panda")
right_panda_arm_instance = plant.GetModelInstanceByName("right::panda")

left_hand_instance = plant.GetModelInstanceByName("left_hand::avatar_gripper_3_model")
right_hand_instance = plant.GetModelInstanceByName("right_hand::avatar_gripper_3f_model")
## Left Arm + Hand
plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0", left_panda_arm_instance),
                RigidTransform(RollPitchYaw(0,0,0), [0.06975, 0.27, 0.5]))

# plant.WeldFrames(plant.GetFrameByName("left_panda_link8"), plant.GetFrameByName("left_gripper_base"),
#                 RigidTransform(RollPitchYaw(0, 3.141592653589793, -0.7853981633974483), [0.0, 0.0, 0.0]))
## Right Arm + Hand
plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0", right_panda_arm_instance),
                RigidTransform(RollPitchYaw(0,0,0), [0.06975, -0.27, 0.5]))

plant.WeldFrames(plant.GetFrameByName("panda_link8", right_panda_arm_instance), plant.GetFrameByName("right_gripper_base", right_hand_instance),
                    RigidTransform(RollPitchYaw(0, 3.141592653589793, -0.7853981633974483), [0.0, 0.0, 0.0]))
                # RigidTransform(RollPitchYaw(1.57, 0, 0), [0.00736, 0.139, -0.066017]))
plant.Finalize()
context = plant.CreateDefaultContext()
# Add the controller
left_controller = builder.AddSystem(PassthroughController(plant=plant, model_instance= left_panda_arm_instance, context = context, side="left"))
right_controller = builder.AddSystem(PassthroughController(plant=plant, model_instance= right_panda_arm_instance, context = context, side="right"))

right_arm_controller = builder.AddSystem(PassthroughController(plant=plant, model_instance= right_hand_instance, context = context, side="right_hand"))

builder.Connect(left_controller.get_output_port(0), plant.get_actuation_input_port(left_panda_arm_instance))
builder.Connect(right_controller.get_output_port(0), plant.get_actuation_input_port(right_panda_arm_instance))

visualizer = MeshcatVisualizer.AddToBuilder(
    builder,
    scene_graph,
    meshcat,  # kIllustration, kProximity
    MeshcatVisualizerParams(role=Role.kIllustration, delete_on_initialization_event=False),
)

diagram = builder.Build()
# diagram.set_name("pick_and_place")


simulator = Simulator(diagram)
context = simulator.get_mutable_context()
simulator.Initialize()
simulator.set_target_realtime_rate(1.0)
# simulator.AdvanceTo(1.0)

# diagram.ForcedPublish(simulator.get_mutable_context())
t = 0
horizon = 100
meshcat.AddButton("Stop Simulation", "Escape")
meshcat.AddButton("Next Step", "Enter")
timestep = 0.01

while meshcat.GetButtonClicks("Stop Simulation") < 1:
    t+= timestep 
    simulator.AdvanceTo(t)
# simulator.AdvanceTo(20)

# RenderDiagram(builder.Build(), max_depth=1)