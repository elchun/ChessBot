import os
import os.path as osp
import sys
import warnings

import numpy as np
from pydrake.all import (
    AbstractValue, Adder, AddMultibodyPlantSceneGraph, BallRpyJoint, BaseField,
    Box, CameraInfo, ClippingRange, CoulombFriction, Cylinder, Demultiplexer,
    DepthImageToPointCloud, DepthRange, DepthRenderCamera, DiagramBuilder,
    FindResourceOrThrow, GeometryInstance, InverseDynamicsController,
    LeafSystem, LoadModelDirectivesFromString,
    MakeMultibodyStateToWsgStateSystem, MakePhongIllustrationProperties,
    MakeRenderEngineVtk, ModelInstanceIndex, MultibodyPlant, Parser,
    PassThrough, PrismaticJoint, ProcessModelDirectives, RenderCameraCore,
    RenderEngineVtkParams, RevoluteJoint, Rgba, RgbdSensor, RigidTransform,
    RollPitchYaw, RotationMatrix, SchunkWsgPositionController, SpatialInertia,
    Sphere, StateInterpolatorWithDiscreteDerivative, UnitInertia)
from pydrake.manipulation.planner import (
    DifferentialInverseKinematicsIntegrator,
    DifferentialInverseKinematicsParameters)

from board import Board

def AddPanda(plant, collide=True):
    """
    Add panda urdf to plant.  Custom function by Ethan, may break.

    Args:
        plant (???): Plant created by drake.

    Returns:
        ???: Index of panda in plant.
    """
    if collide:
        sdf_path = FindResourceOrThrow(
        "drake/manipulation/models/"
        # "franka_description/urdf/panda_arm_hand.urdf")
        "franka_description/urdf/panda_arm.urdf")
    else:
        sdf_path = FindResourceOrThrow(
        "drake/manipulation/models/"
        # "franka_description/urdf/panda_arm_hand_no_collide.urdf")
        "franka_description/urdf/panda_arm_no_collide.urdf")


    parser = Parser(plant)
    panda = parser.AddModelFromFile(sdf_path)

    panda_base_frame = plant.GetFrameByName('panda_link0')
    panda_transform = RigidTransform(
    RollPitchYaw(np.asarray([0, 0, -np.pi/2])), p=[0, 0.45, 0])

    plant.WeldFrames(plant.world_frame(), panda_base_frame, panda_transform)

    # Set default positions:
    # q0 = [0.0, 0.0, 0, -1.2, 0, 1.6, 0]
    # q0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    q0 = [0.0, 0, 0.0, -np.pi/2, 0.0, np.pi/2, np.pi/4]
    index = 0
    for joint_index in plant.GetJointIndices(panda):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1

    return panda


def AddBoard(plant):
    board = Board()

    parser = Parser(plant)
    board_idx = parser.AddModelFromFile(osp.join(board.model_dir, board.board_fn))

    idx_to_location = {}
    for location, piece in board.starting_board.items():
        name = location + piece  # Very arbitrary, may change later
        idx = parser.AddModelFromFile(osp.join(board.model_dir, board.piece_to_fn[piece]), name)

        idx_to_location[idx] = location

    # Weld the table to the world so that it's fixed during the simulation.
    board_frame = plant.GetFrameByName("board_body")
    plant.WeldFrames(plant.world_frame(), board_frame)

    return board, board_idx, idx_to_location


def SetBoard(plant, idx_to_location, board):
    board_piece_offset = 0.0
    plant_context = plant.CreateDefaultContext()

    board_frame = plant.GetFrameByName("board_body")
    X_WorldBoard= board_frame.CalcPoseInWorld(plant_context)

    for idx, location in idx_to_location.items():
        piece = plant.GetBodyByName("piece_body", idx)
        # benchy = plant.GetBodyByName("benchy_body", name)
        x, y = board.get_xy_location(location)
        X_BoardPiece = RigidTransform(
            RollPitchYaw(np.asarray([0, 0, 0])), p=[x, y, board_piece_offset])
        X_BoardPiece = X_WorldBoard.multiply(X_BoardPiece)
        plant.SetDefaultFreeBodyPose(piece, X_BoardPiece)


def AddRgbdSensor(builder,
                  scene_graph,
                  X_PC,
                  depth_camera=None,
                  renderer=None,
                  parent_frame_id=None):
    """
    Adds a RgbdSensor to to the scene_graph at (fixed) pose X_PC relative to
    the parent_frame.  If depth_camera is None, then a default camera info will
    be used.  If renderer is None, then we will assume the name 'my_renderer',
    and create a VTK renderer if a renderer of that name doesn't exist.  If
    parent_frame is None, then the world frame is used.
    """
    if sys.platform == "linux" and os.getenv("DISPLAY") is None:
        from pyvirtualdisplay import Display
        virtual_display = Display(visible=0, size=(1400, 900))
        virtual_display.start()

    if not renderer:
        renderer = "my_renderer"

    if not parent_frame_id:
        parent_frame_id = scene_graph.world_frame_id()

    if not scene_graph.HasRenderer(renderer):
        scene_graph.AddRenderer(renderer,
                                MakeRenderEngineVtk(RenderEngineVtkParams()))

    if not depth_camera:
        depth_camera = DepthRenderCamera(
            RenderCameraCore(
                renderer, CameraInfo(width=640, height=480, fov_y=np.pi / 4.0),
                ClippingRange(near=0.1, far=10.0), RigidTransform()),
            DepthRange(0.1, 10.0))

    rgbd = builder.AddSystem(
        RgbdSensor(parent_id=parent_frame_id,
                   X_PB=X_PC,
                   depth_camera=depth_camera,
                   show_window=False))

    builder.Connect(scene_graph.get_query_output_port(),
                    rgbd.query_object_input_port())

    return rgbd


def AddRgbdSensors(builder,
                   plant,
                   scene_graph,
                   also_add_point_clouds=True,
                   model_instance_prefix="camera",
                   depth_camera=None,
                   renderer=None):
    """
    Adds a RgbdSensor to the first body in the plant for every model instance
    with a name starting with model_instance_prefix.  If depth_camera is None,
    then a default camera info will be used.  If renderer is None, then we will
    assume the name 'my_renderer', and create a VTK renderer if a renderer of
    that name doesn't exist.
    """
    if sys.platform == "linux" and os.getenv("DISPLAY") is None:
        from pyvirtualdisplay import Display
        virtual_display = Display(visible=0, size=(1400, 900))
        virtual_display.start()

    if not renderer:
        renderer = "my_renderer"

    if not scene_graph.HasRenderer(renderer):
        scene_graph.AddRenderer(renderer,
                                MakeRenderEngineVtk(RenderEngineVtkParams()))

    if not depth_camera:
        depth_camera = DepthRenderCamera(
            RenderCameraCore(
                renderer, CameraInfo(width=640, height=480, fov_y=np.pi / 4.0),
                ClippingRange(near=0.1, far=10.0), RigidTransform()),
            DepthRange(0.1, 10.0))

    for index in range(plant.num_model_instances()):
        model_instance_index = ModelInstanceIndex(index)
        model_name = plant.GetModelInstanceName(model_instance_index)

        if model_name.startswith(model_instance_prefix):
            body_index = plant.GetBodyIndices(model_instance_index)[0]
            rgbd = builder.AddSystem(
                RgbdSensor(parent_id=plant.GetBodyFrameIdOrThrow(body_index),
                           X_PB=RigidTransform(),
                           depth_camera=depth_camera,
                           show_window=False))
            rgbd.set_name(model_name)

            builder.Connect(scene_graph.get_query_output_port(),
                            rgbd.query_object_input_port())

            # Export the camera outputs
            builder.ExportOutput(rgbd.color_image_output_port(),
                                 f"{model_name}_rgb_image")
            builder.ExportOutput(rgbd.depth_image_32F_output_port(),
                                 f"{model_name}_depth_image")
            builder.ExportOutput(rgbd.label_image_output_port(),
                                 f"{model_name}_label_image")

            if also_add_point_clouds:
                # Add a system to convert the camera output into a point cloud
                to_point_cloud = builder.AddSystem(
                    DepthImageToPointCloud(camera_info=rgbd.depth_camera_info(),
                                           fields=BaseField.kXYZs
                                           | BaseField.kRGBs))
                builder.Connect(rgbd.depth_image_32F_output_port(),
                                to_point_cloud.depth_image_input_port())
                builder.Connect(rgbd.color_image_output_port(),
                                to_point_cloud.color_image_input_port())

                class ExtractBodyPose(LeafSystem):

                    def __init__(self, body_index):
                        LeafSystem.__init__(self)
                        self.body_index = body_index
                        self.DeclareAbstractInputPort(
                            "poses",
                            plant.get_body_poses_output_port().Allocate())
                        self.DeclareAbstractOutputPort(
                            "pose",
                            lambda: AbstractValue.Make(RigidTransform()),
                            self.CalcOutput)

                    def CalcOutput(self, context, output):
                        poses = self.EvalAbstractInput(context, 0).get_value()
                        pose = poses[int(self.body_index)]
                        output.get_mutable_value().set(pose.rotation(),
                                                       pose.translation())

                camera_pose = builder.AddSystem(ExtractBodyPose(body_index))
                builder.Connect(plant.get_body_poses_output_port(),
                                camera_pose.get_input_port())
                builder.Connect(camera_pose.get_output_port(),
                                to_point_cloud.GetInputPort("camera_pose"))

                # Export the point cloud output.
                builder.ExportOutput(to_point_cloud.point_cloud_output_port(),
                                     f"{model_name}_point_cloud")


def AddPandaDifferentialIK(builder, plant, frame=None):
    params = DifferentialInverseKinematicsParameters(plant.num_positions(),
                                                     plant.num_velocities())
    time_step = plant.time_step()
    q0 = plant.GetPositions(plant.CreateDefaultContext())
    params.set_nominal_joint_position(q0)
    params.set_end_effector_angular_speed_limit(2)
    params.set_end_effector_translational_velocity_limits([-2, -2, -2],
                                                          [2, 2, 2])
    panda_velocity_limits = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])
    params.set_joint_velocity_limits(
        (-panda_velocity_limits, panda_velocity_limits))
    params.set_joint_centering_gain(10 * np.eye(7))
    if frame is None:
        frame = plant.GetFrameByName("body")
    differential_ik = builder.AddSystem(
        DifferentialInverseKinematicsIntegrator(
            plant,
            frame,
            time_step,
            params,
            log_only_when_result_state_changes=True))
    return differential_ik

# TODO: take argument for whether we want the welded fingers version or not
def AddWsgPanda(plant,
           panda_model_instance,
           roll=np.pi / 2.0,
           sphere=False):
    parser = Parser(plant)
    gripper = parser.AddModelFromFile(
        FindResourceOrThrow(
            "drake/manipulation/models/"
            "wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf"))

    X_7G = RigidTransform(RollPitchYaw(np.pi / 2.0, 0, roll), [0, 0, 0.09])
    plant.WeldFrames(plant.GetFrameByName("panda_link8", panda_model_instance),
                     plant.GetFrameByName("body", gripper), X_7G)
    return gripper


def MakeChessManipulationStation(time_step=0.002,
                                 panda_prefix='panda',
                                 camera_prefix='camera'):
    """
    Create Manipulation station with panda.  Hevaily based on MakeManipulationStation.

    Args:
        time_step (float, optional): _description_. Defaults to 0.002.
        panda_prefix (str, optional): _description_. Defaults to 'panda'.
        camera_prefix (str, optional): _description_. Defaults to 'camera'.

    Returns:
        _type_: _description_
    """
    builder = DiagramBuilder()

    plant, scene_graph = AddMultibodyPlantSceneGraph(builder,
                                                     time_step=time_step)

    parser = Parser(plant)
    AddPanda(plant)
    board, board_idx, idx_to_location = AddBoard(plant)

    plant.set_stiction_tolerance(0.001)

    plant.Finalize()

    SetBoard(plant, idx_to_location, board)

    # print(plant)

    for i in range(plant.num_model_instances()):
        model_instance = ModelInstanceIndex(i)
        model_instance_name = plant.GetModelInstanceName(model_instance)
        # print('name: ', model_instance_name)
        if model_instance_name.startswith(panda_prefix):
            num_panda_positions = plant.num_positions(model_instance)

            # I need a PassThrough system so that I can export the input port.
            panda_position = builder.AddSystem(PassThrough(num_panda_positions))
            builder.ExportInput(panda_position.get_input_port(),
                                model_instance_name + "_position")
            builder.ExportOutput(panda_position.get_output_port(),
                                 model_instance_name + "_position_commanded")
            # Export the iiwa "state" outputs.
            demux = builder.AddSystem(
                Demultiplexer(2 * num_panda_positions, num_panda_positions))
            builder.Connect(plant.get_state_output_port(model_instance),
                            demux.get_input_port())
            builder.ExportOutput(demux.get_output_port(0),
                                 model_instance_name + "_position_measured")
            builder.ExportOutput(demux.get_output_port(1),
                                 model_instance_name + "_velocity_estimated")
            builder.ExportOutput(plant.get_state_output_port(model_instance),
                                 model_instance_name + "_state_estimated")

            # Make the plant for the iiwa controller to use.
            controller_plant = MultibodyPlant(time_step=time_step)
            controller_panda = AddPanda(controller_plant, collide=False)
            controller_plant.Finalize()

            panda_controller = builder.AddSystem(
                InverseDynamicsController(controller_plant,
                                          kp=[100] * num_panda_positions,
                                          ki=[1] * num_panda_positions,
                                          kd=[20] * num_panda_positions,
                                          has_reference_acceleration=False))
            panda_controller.set_name(model_instance_name + "_controller")
            builder.Connect(plant.get_state_output_port(model_instance),
                            panda_controller.get_input_port_estimated_state())


            # Add in the feed-forward torque
            adder = builder.AddSystem(Adder(2, num_panda_positions))
            builder.Connect(panda_controller.get_output_port_control(),
                            adder.get_input_port(0))
            # Use a PassThrough to make the port optional (it will provide zero
            # values if not connected).
            torque_passthrough = builder.AddSystem(
                PassThrough([0] * num_panda_positions))
            builder.Connect(torque_passthrough.get_output_port(),
                            adder.get_input_port(1))
            builder.ExportInput(torque_passthrough.get_input_port(),
                                model_instance_name + "_feedforward_torque")
            builder.Connect(adder.get_output_port(),
                            plant.get_actuation_input_port(model_instance))

            # Add discrete derivative to command velocities.
            desired_state_from_position = builder.AddSystem(
                StateInterpolatorWithDiscreteDerivative(
                    num_panda_positions,
                    time_step,
                    suppress_initial_transient=True))
            desired_state_from_position.set_name(
                model_instance_name + "_desired_state_from_position")
            builder.Connect(desired_state_from_position.get_output_port(),
                            panda_controller.get_input_port_desired_state())
            builder.Connect(panda_position.get_output_port(),
                            desired_state_from_position.get_input_port())

            # Export commanded torques.
            builder.ExportOutput(adder.get_output_port(),
                                 model_instance_name + "_torque_commanded")
            builder.ExportOutput(adder.get_output_port(),
                                 model_instance_name + "_torque_measured")

            builder.ExportOutput(
                plant.get_generalized_contact_forces_output_port(
                    model_instance), model_instance_name + "_torque_external")

    # Cameras.
    AddRgbdSensors(builder,
                   plant,
                   scene_graph,
                   model_instance_prefix=camera_prefix)

    # Export "cheat" ports
    builder.ExportOutput(scene_graph.get_query_output_port(), "query_object")
    builder.ExportOutput(plant.get_contact_results_output_port(),
                         "contact_results")
    builder.ExportOutput(plant.get_state_output_port(),
                         "plant_continuous_state")
    builder.ExportOutput(plant.get_body_poses_output_port(), "body_poses")

    diagram = builder.Build()
    diagram.set_name("ManipulationStation")
    return diagram


if __name__ == '__main__':
    MakeChessManipulationStation()





