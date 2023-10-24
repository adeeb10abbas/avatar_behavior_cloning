import numpy as np
from pydrake.all import *
from utils.geometry_utils import *
import matplotlib.pyplot as plt

def vis_hand_pose(meshcat, X_TG, path, load=False, color=[0.1, 0.1, 0.1, 0.3]):
    """render the tool at the pose tf in meshcat"""
    if type(X_TG) is not RigidTransform:
        X_TG = RigidTransform(X_TG @ rotX(-np.pi / 2))
    if meshcat is None:
        return

    if load:
        mesh = ReadObjToTriangleSurfaceMesh("models/hand_finger.obj")
        meshcat.Delete(path)  # delete the old one
        meshcat.SetObject(path, mesh, Rgba(*color))
    meshcat.SetProperty(path, "color", color)
    meshcat.SetTransform(path, X_TG)


def _get_geometries_direct(plant, scene_graph, bodies):
    geometry_ids = []
    inspector = scene_graph.model_inspector()
    for geometry_id in inspector.GetAllGeometryIds():
        body = plant.GetBodyFromFrameId(inspector.GetFrameId(geometry_id))
        if body in bodies:
            geometry_ids.append(geometry_id)
    geometry_ids.sort()
    return geometry_ids


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


def get_bodies(plant, model_instances=None):

    # TODO(eric.cousineau): Hoist this somewhere?

    return _get_plant_aggregate(plant.num_bodies, plant.get_body, BodyIndex, model_instances)


def get_geometries(plant, scene_graph, bodies=None):

    """Returns all GeometryId's attached to bodies. Assumes corresponding

    FrameId's have been added."""

    if bodies is None:

        bodies = get_bodies(plant)

    geometry_ids = _get_geometries_direct(plant, scene_graph, list(bodies))

    return sorted(geometry_ids, key=lambda x: x.get_value())


def filter_all_collisions(plant, scene_graph):

    bodies = get_bodies(plant)

    geometries = get_geometries(plant, scene_graph, bodies)

    filter_manager = scene_graph.collision_filter_manager()

    geometry_set = GeometrySet(geometries)

    declaration = CollisionFilterDeclaration()

    declaration.ExcludeWithin(geometry_set)

    filter_manager.Apply(declaration)


def remove_joint_limits(plant):

    # TODO(eric.cousineau): Handle actuator limits when Drake supports mutating

    # them.

    for joint in get_joints(plant):

        num_q = joint.num_positions()

        num_v = joint.num_velocities()

        joint.set_position_limits(np.full(num_q, -np.inf), np.full(num_q, np.inf))

        joint.set_velocity_limits(np.full(num_v, -np.inf), np.full(num_v, np.inf))

        joint.set_acceleration_limits(np.full(num_v, -np.inf), np.full(num_v, np.inf))


def remove_joint_damping(plant, model_instances=None):

    count = 0

    for joint in get_joints(plant, model_instances=model_instances):

        if isinstance(joint, (RevoluteJoint, PrismaticJoint)):

            # joint.set_default_damping(0.0)

            count += 1

    return count


def simplify_plant(plant, scene_graph):

    """

    Zeros out gravity, removes collisions, and effectively disables joint

    limits.

    """

    plant.mutable_gravity_field().set_gravity_vector(np.zeros(3))

    # filter_all_collisions(plant, scene_graph)

    remove_joint_damping(plant)

    # remove_joint_limits(plant)


def start_logger():
    plt.close()
    fig,  ax = plt.subplots()

    # fit the labels inside the plot
    ax.set_xlim(0, 25)
    ax.set_ylim(-0.5, 0.5)
    (state_plot_command,) = plt.plot([], [], label="commanded")
    (state_plot_actual,) = plt.plot([], [], label="actual")
    plt.legend()
    plt.show(block=False)
    return state_plot_actual, state_plot_command

def update_line(state_plot_actual, state_plot_command, new_data_actual, new_data_command):
    state_plot_actual.set_xdata(np.append( state_plot_actual.get_xdata(), new_data_actual[0]))
    state_plot_actual.set_ydata(np.append( state_plot_actual.get_ydata(), new_data_actual[1]))
    state_plot_command.set_xdata(np.append( state_plot_command.get_xdata(), new_data_command[0]))
    state_plot_command.set_ydata(np.append( state_plot_command.get_ydata(), new_data_command[1]))

    plt.draw()
    plt.autoscale()
    plt.pause(0.001)