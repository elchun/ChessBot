# Script to simulate and parse a board
# Loosely based on https://github.com/RussTedrake/manipulation/blob/master/segmentation_data.py

import argparse
import json
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
from PIL import Image
import os
import os.path as osp
import sys
import shutil
import warnings
import random

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import torchvision.transforms.functional as Tf

from scipy import ndimage

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    FindResourceOrThrow,
    Parser,
    RandomGenerator,
    RigidTransform,
    Role,
    RollPitchYaw,
    Simulator,
    UniformlyRandomRotationMatrix,
    MakeRenderEngineVtk,
    RenderEngineVtkParams,
    RenderCameraCore,
    DepthRenderCamera,
    PointCloud,
    BaseField,
    Fields,
)

from pydrake.all import (
    AbstractValue, Adder, AddMultibodyPlantSceneGraph, BallRpyJoint, BaseField,
    Box, CameraInfo, ClippingRange,
    DepthImageToPointCloud, DepthRange, DepthRenderCamera, DiagramBuilder,
    FindResourceOrThrow,
    LeafSystem,
    MakeRenderEngineVtk, ModelInstanceIndex, Parser, RenderCameraCore,
    RenderEngineVtkParams, RgbdSensor, RigidTransform,
    RollPitchYaw, Role,
    ImageRgba8U, ImageDepth32F)

from chess_bot.utils.utils import colorize_labels

from chess_bot.resources.board import Board
from chess_bot.utils.path_util import get_chessbot_src
from chess_bot.perception.extract_masks import ExtractMasks
from chess_bot.perception.create_pointclouds import CreatePointclouds
from chess_bot.perception.icp_system import ICPSystem

rng = np.random.default_rng()  # this is for python
generator = RandomGenerator(rng.integers(1000))  # for c++

class GameStation():
    def __init__(self):
        """
        Class that simulates a chess board to generate data for Mask R-CNN
        """
        self.station = self.make_data_station()
        self.station_context = self.station.CreateDefaultContext()

        self.simulator = Simulator(self.station)
        self.simulator_context = self.simulator.get_mutable_context()
        self.simulator.AdvanceTo(0.01)

        # self.sim_time = 0.0


    def make_data_station(self, time_step=0.002):
        """
        Helper function to create the diagram for a chess board

        Args:
            time_step (float, optional): Timestep of plant. Defaults to 0.002.

        Returns:
            scene diagram?: Result of builder.Build().
        """
        builder = DiagramBuilder()

        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(builder,
                                                     time_step=time_step)

        self.inspector = self.scene_graph.model_inspector()
        self.parser = Parser(self.plant)

        self.board, board_idx, self.idx_to_location, self.instance_id_to_class_name = self._add_board()

        self.plant.set_stiction_tolerance(0.001)

        # Optimized for 1080 x 1920
        # X_Camera = RigidTransform(RollPitchYaw(-np.pi/2 + -0.61, 0, -np.pi/2), [-0.65, 0, 0.44])
        # X_Camera = RigidTransform(RollPitchYaw(-np.pi/2 + -0.75, 0, -np.pi/2), [-0.65, 0.2667, 0.6])  # for testing position
        X_Camera = RigidTransform(RollPitchYaw(-np.pi/2 + -0.77, 0, -np.pi/2), [-0.63, 0, 0.6])
        # X_Camera = RigidTransform(RollPitchYaw(np.pi/6, 0, -np.pi/2), [-0.6, 0, 0.4])
        camerabox_fn = osp.join(get_chessbot_src(), 'resources/models/camera_box.sdf')
        camera_instance = self.parser.AddModelFromFile(camerabox_fn, 'camera')
        camera_frame = self.plant.GetFrameByName('base', camera_instance)
        self.plant.WeldFrames(self.plant.world_frame(), camera_frame, X_Camera)
        # AddMultibodyTriad(camera_frame, scene_graph, length=0.1, radius=0.005)

        self.plant.Finalize()

        self._set_default_board()

        # Cameras and perception system
        camera_prefix = 'camera'
        port_dict = AddRgbdSensors(builder,
                    self.plant,
                    self.scene_graph,
                    model_instance_prefix=camera_prefix)

        pcd_port_0 = port_dict['pcd_ports'][0]

        # Add ICP System to first pcd port
        icp = builder.AddSystem(ICPSystem())
        builder.Connect(pcd_port_0.GetOutputPort('pcd_stack'),
            icp.GetInputPort('raw_pcd_stack'))

        builder.ExportOutput(icp.GetOutputPort('icp_pcd_stack'),
            'icp_pcd_stack')

        diagram = builder.Build()
        diagram.set_name("ManipulationStation")
        return diagram

    def read_rgbd_sensor(self):
        """
        Get color, depth, and label images for the board

        Returns:
            tuple(array): color image, depth image, label image
        """
        color_image = self.station.GetOutputPort("camera_rgb_image").Eval(self.station_context).data
        depth_image = self.station.GetOutputPort("camera_depth_image").Eval(self.station_context).data
        label_image = self.station.GetOutputPort("camera_depth_image").Eval(self.station_context).data
        return color_image, depth_image, label_image

    def show_label_image(self):
        """
        Call this in a ipynb window.  Shows the output of the mask camera image.
        """
        label_image = self.station.GetOutputPort("camera_label_image").Eval(self.station_context)
        plt.imshow(colorize_labels(label_image.data))
        print('Num Unique values: ', len(np.unique(label_image.data)))

    def show_rgb_image(self):
        """
        Call this in a ipynb window.  Shows the output of the rgb camera image.
        """
        color_image = self.station.GetOutputPort("camera_rgb_image").Eval(self.station_context)
        plt.imshow(color_image.data)

    def get_processed_pcds(self):
        """
        Returns list of processed point clouds of pieces, and piece labels
        """
        return self.station.GetOutputPort('icp_pcd_stack').Eval(self.station_context)

    def _add_board(self):
        """
        Helper functoin to create and add a board to the station.

        Returns:
            board, board_idx, idx_to_location, instance_id_to_class_name: Extra
                information generated when the board is created.
        """
        board = Board()

        parser = Parser(self.plant)
        board_idx = parser.AddModelFromFile(osp.join(board.model_dir, board.board_fn))

        idx_to_location = {}

        instance_id_to_class_name = {}

        for location, piece in board.starting_board.items():
            name = location + piece  # Very arbitrary, may change later
            idx = parser.AddModelFromFile(osp.join(board.model_dir, board.piece_to_fn[piece]), name)

            # -- Add labels for label generation -- #
            frame_id = self.plant.GetBodyFrameIdOrThrow(
                self.plant.GetBodyIndices(idx)[0])
            geometry_ids = self.inspector.GetGeometries(frame_id, Role.kPerception)

            for geom_id in geometry_ids:
                instance_id_to_class_name[int(
                    self.inspector.GetPerceptionProperties(geom_id).GetProperty(
                        "label", "id"))] = name

            idx_to_location[idx] = location

        # Weld the table to the world so that it's fixed during the simulation.
        board_frame = self.plant.GetFrameByName("board_body")
        # plant.WeldFrames(plant.world_frame(), board_frame, RigidTransform(RollPitchYaw(np.array([0, 0, np.pi/2])), [0, 0, 0]))
        self.plant.WeldFrames(self.plant.world_frame(), board_frame)

        # print(instance_id_to_class_name)
        return board, board_idx, idx_to_location, instance_id_to_class_name

    def _set_default_board(self):
        """
        Move the pieces to the default positions (start of chess game).
        """
        board_piece_offset = 0.0
        plant_context = self.plant.CreateDefaultContext()

        board_frame = self.plant.GetFrameByName("board_body")
        X_WorldBoard= board_frame.CalcPoseInWorld(plant_context)


        for idx, location in self.idx_to_location.items():
            piece = self.plant.GetBodyByName("piece_body", idx)
            x, y = self.board.get_xy_location(location)

            # Flip white pieces to face the right way
            if y < 0:
                yaw = np.pi
            else:
                yaw = 0

            X_BoardPiece = RigidTransform(
                RollPitchYaw(np.asarray([0, 0, yaw])), p=[x, y, board_piece_offset])
            X_BoardPiece = X_WorldBoard.multiply(X_BoardPiece)
            self.plant.SetDefaultFreeBodyPose(piece, X_BoardPiece)

    def set_arbitrary_board(self, num_pieces=10):
        """
        Move <num_pieces> pieces to random locations on the board.  Make sure
        that no pieces are on top of eachother.

        Args:
            num_pieces (int, optional): number of pieces to show. Defaults to 10.
        """
        board_piece_offset = 0.0
        plant_context = self.plant.GetMyMutableContextFromRoot(self.station_context)

        board_frame = self.plant.GetFrameByName("board_body")
        X_WorldBoard= board_frame.CalcPoseInWorld(plant_context)

        locations = set()  # set of (x, y) tuples

        indices = list(self.idx_to_location.keys())
        random.shuffle(indices)

        for idx in indices[:num_pieces]:
            piece = self.plant.GetBodyByName("piece_body", idx)
            # benchy = plant.GetBodyByName("benchy_body", name)

            # Find a position that is not currently occupied
            while True:
                x = np.random.randint(0, 8)
                y = np.random.randint(0, 8)
                if (x, y) not in locations:
                    break
            locations.add((x, y))
            # print('x: ', x, 'y: ', y)
            x, y = self.board.get_xy_location_from_idx(x, y)
            yaw = np.random.random() * 2 * np.pi
            X_BoardPiece = RigidTransform(
                RollPitchYaw(np.asarray([0, 0, yaw])), p=[x, y, board_piece_offset])
            X_BoardPiece = X_WorldBoard.multiply(X_BoardPiece)
            self.plant.SetFreeBodyPose(plant_context, piece, X_BoardPiece)

        # Put pieces off screen
        for i, idx in enumerate(indices[num_pieces:]):
            piece = self.plant.GetBodyByName("piece_body", idx)
            x, y = 1, 1 + i
            X_BoardPiece = RigidTransform(
                RollPitchYaw(np.asarray([0, 0, 0])), p=[x, y, board_piece_offset])
            X_BoardPiece = X_WorldBoard.multiply(X_BoardPiece)
            self.plant.SetFreeBodyPose(plant_context, piece, X_BoardPiece)
        # self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 2.0)
        # print('sim_time: ', self.simulator.get_context().get_time())


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

    Rn this only works for adding one camera because the icp stuff won't merge more
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
                # renderer, CameraInfo(width=640, height=480, fov_y=np.pi / 4.0),
                # ClippingRange(near=0.1, far=10.0), RigidTransform()),
                renderer, CameraInfo(width=1920, height=1080, fov_y=np.pi / 6.0),
                ClippingRange(near=0.1, far=10.0), RigidTransform()),
            DepthRange(0.1, 10.0))

    pcd_ports = []

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


                masks = builder.AddSystem(ExtractMasks(rgbd))
                builder.Connect(rgbd.color_image_output_port(),
                    masks.GetInputPort('rgb_image'))
                builder.Connect(rgbd.depth_image_32F_output_port(),
                    masks.GetInputPort('depth_image'))
                builder.ExportOutput(masks.GetOutputPort('masked_depth_image'),
                    f"{model_name}_masked_depth_image")

                pcds = builder.AddSystem(CreatePointclouds(rgbd))
                builder.Connect(masks.GetOutputPort('masked_depth_image'),
                    pcds.GetInputPort('depth_image_stack'))
                builder.Connect(rgbd.body_pose_in_world_output_port(),
                    pcds.GetInputPort('rgbd_sensor_body_pose'))
                builder.ExportOutput(pcds.GetOutputPort('pcd_stack'),
                    f"{model_name}_pcd_stack")

                # icp = builder.AddSystem(CreatePointclouds(rgbd))
                # icp = builder.AddSystem(ICPSystem())
                # builder.Connect(pcd_port_0.GetOutputPort('pcd_stack'),
                #     icp.GetInputPort('raw_pcd_stack'))

                # builder.ExportOutput(icp.GetOutputPort('icp_pcd_stack'),
                #     'icp_pcd_stack')

                pcd_ports.append(pcds)

    ports = {
        'pcd_ports': pcd_ports,
    }

    return ports