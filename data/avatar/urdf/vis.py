## Small script to visualize a URDF model in Meshcat. To be used as a quick utility.
import argparse
import numpy as np
import os
from pydrake.geometry import StartMeshcat
from pydrake.visualization import ModelVisualizer
from pydrake.multibody.plant import AddMultibodyPlant, MultibodyPlantConfig, MultibodyPlant
from pydrake.systems.framework import DiagramBuilder, OutputPort
from pydrake.multibody.parsing import Parser
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.visualization import AddDefaultVisualization, ModelVisualizer
from pydrake.systems.analysis import Simulator
from pydrake.systems.primitives import ConstantVectorSource
import dataclasses as dc

meshcat = StartMeshcat()

@dc.dataclass
class DofSetForDynamics:
    """
    Like DofSet, but for dynamics where we are not constrained to nq = nv = nu.
    """

    q: np.ndarray  # bool
    v: np.ndarray  # bool
    u: np.ndarray  # bool

def make_empty_dofset(plant):
    return DofSetForDynamics(
        q=np.zeros(plant.num_positions(), dtype=bool),
        v=np.zeros(plant.num_velocities(), dtype=bool),
        u=np.zeros(plant.num_actuated_dofs(), dtype=bool),
    )
# def create_scene(sim_time_step: int):
#     meshcat.Delete()
#     meshcat.DeleteAddedControls()
def articulated_model_instance_dofset(plant, instance):
    """
    Returns DofSetForDynamics given a particular model instance.
    """
    dofs = make_empty_dofset(plant)
    joint_indices = plant.GetJointIndices(instance)
    for joint_index in joint_indices:
        joint = plant.get_joint(joint_index)
        start_in_q = joint.position_start()
        end_in_q = start_in_q + joint.num_positions()
        dofs.q[start_in_q:end_in_q] = True
        start_in_v = joint.velocity_start()
        end_in_v = start_in_v + joint.num_velocities()
        dofs.v[start_in_v:end_in_v] = True

    B_full = plant.MakeActuationMatrix()
    B = B_full[dofs.v]
    assert set(B.flat) <= {0.0, 1.0}
    B_rows = B.sum(axis=0)
    assert set(B_rows.flat) <= {0.0, 1.0}
    dofs.u[:] = B_rows.astype(bool)

    nq = plant.num_positions(instance)
    nv = plant.num_velocities(instance)
    nu = plant.num_actuated_dofs(instance)
    assert dofs.q.sum() == nq
    assert dofs.v.sum() == nv
    assert dofs.u.sum() == nu
    return dofs


builder = DiagramBuilder()
multibody_plant_config = MultibodyPlantConfig(time_step=0.001, discrete_contact_solver="sap")
plant, scene_graph = AddMultibodyPlant(config=multibody_plant_config, builder=builder)
parser = Parser(plant)



## Load in the URDFs 
left_panda_model_urdf = "avatar_left_arm.urdf"
right_panda_model_urdf = "avatar_right_arm.urdf"

## Ideally it would be two arms as well 
left_hand_model_urdf = "avatar_gripper_left.urdf"
right_hand_model_urdf = "avatar_gripper_right_nomimic.urdf"
## 
# table_top_model = "env_objects/table_top.urdf"
# and maybe some other jazz

#base 
base_model_urdf = "avatar_base.urdf"

parser = Parser(plant=plant, scene_graph=scene_graph)
left_parser = Parser(plant=plant, scene_graph=scene_graph)
right_parser = Parser(plant=plant, scene_graph=scene_graph)

# Let's load all the stuff we have
# Base
base_instance = parser.AddModelFromFile(base_model_urdf)
# print("Number of actuators after adding base: ", plant.num_actuators())

## Right Arm Stuff
right_arm_instance = parser.AddModelFromFile(right_panda_model_urdf)
right_hand_instance = parser.AddModelFromFile(right_hand_model_urdf)
# print("Number of actuators after adding left arm: ", plant.num_actuated_dofs())

## Left Arm Stuff
left_arm_instance = parser.AddModelFromFile(left_panda_model_urdf)
left_hand_instance = parser.AddModelFromFile(left_hand_model_urdf)
# print("Number of actuators after adding right arm: ", plant.num_actuators())

## Weld everything together and visualize! 
plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base"),
                RigidTransform(RollPitchYaw(0,0,0), [0.5, 0.2, 0.5]))

## Left Arm + Hand
plant.WeldFrames(plant.GetFrameByName("base"), plant.GetFrameByName("left_panda_link0"),
                RigidTransform(RollPitchYaw(0,0,0), [0.06975, 0.27, 0.01]))

plant.WeldFrames(plant.GetFrameByName("left_panda_link8"), plant.GetFrameByName("left_gripper_base"),
                RigidTransform(RollPitchYaw(0, 3.141592653589793, -0.7853981633974483), [0.0, 0.0, 0.0]))
## Right Arm + Hand
plant.WeldFrames(plant.GetFrameByName("base"), plant.GetFrameByName("right_panda_link0"),
                RigidTransform(RollPitchYaw(0,0,0), [0.06975, -0.27, 0.01]))

plant.WeldFrames(plant.GetFrameByName("right_panda_link8"), plant.GetFrameByName("right_gripper_base"),
                RigidTransform(RollPitchYaw(1.57, 0, 0), [0.00736, 0.139, -0.066017]))
plant.Finalize()

plant_context = plant.CreateDefaultContext()
AddDefaultVisualization(builder=builder, meshcat=meshcat)
dig = builder.Build()

# Right Arm + Hand
dofs = articulated_model_instance_dofset(plant, right_hand_instance)
print("dofs.u", dofs.u.sum())
# assert dofs.u.sum() == 7
print("dofs.v", dofs.v.sum())
# assert dofs.v.sum() == 7
print("dofs.q", dofs.q.sum())
# assert dofs.q.sum() == 7



