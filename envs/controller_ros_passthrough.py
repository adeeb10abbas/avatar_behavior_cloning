
import mpld3
import numpy as np
from matplotlib import pyplot as plt
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    AngleAxis,
    DiagramBuilder,
    Integrator,
    JacobianWrtVariable,
    LeafSystem,
    MeshcatVisualizer,
    MultibodyPlant,
    MultibodyPositionToGeometryPose,
    Parser,
    PiecewisePolynomial,
    PiecewisePose,
    Quaternion,
    Rgba,
    RigidTransform,
    RotationMatrix,
    SceneGraph,
    Simulator,
    StartMeshcat,
    TrajectorySource,
    BasicVector,
)
import numpy as np
from pydrake.systems.framework import Context, SystemOutput

class PassthroughController(LeafSystem):
    def __init__(self, plant, model_instance, context, side: str):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = context
        if "hand" not in side:
            if side == "left":
                self._iiwa = plant.GetModelInstanceByName("left::panda")
            elif side == "right":
                self._iiwa = plant.GetModelInstanceByName("right::panda")
            self.DeclareVectorOutputPort("panda.position", 7, self.CalcOutput)
            self.iiwa_start = self._plant.GetJointByName("panda_joint1", self._iiwa).position_start()
            self.iiwa_end = self._plant.GetJointByName("panda_joint8", self._iiwa).position_start()

        else:
            if side == "left":
                self._iiwa = plant.GetModelInstanceByName("left_hand::avatar_gripper_3_model")
            elif side == "right":
                self._iiwa = plant.GetModelInstanceByName("right_hand::avatar_gripper_3f_model")
            
            self.DeclareVectorOutputPort("gripper.position", 1, self.CalcOutput)
            self.iiwa_start = self._plant.GetJointByName("left_gripper_finger1", self._iiwa).position_start()
            self.iiwa_end = self._plant.GetJointByName("left_gripper_finger3", self._iiwa).position_start()

    def CalcOutput(self, context, output) -> None:
        output.SetFromVector(np.zeros(7))