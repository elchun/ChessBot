# Utility functions for game_station.  Based on telop example, among other things

import numpy as np
import os
import os.path as osp
import sys

from pydrake.all import (
    RigidTransform,
    AbstractValue,
    BaseField,
    CameraInfo,
    ClippingRange,
    DepthImageToPointCloud,
    DepthRange,
    DepthRenderCamera,
    FindResourceOrThrow,
    LeafSystem,
    MakeRenderEngineVtk,
    ModelInstanceIndex,
    Parser,
    RenderCameraCore,
    RenderEngineVtkParams,
    RevoluteJoint,
    RgbdSensor,
    RollPitchYaw)

from pydrake.manipulation.planner import (
    DifferentialInverseKinematicsIntegrator,
    DifferentialInverseKinematicsParameters)


from chess_bot.perception.extract_masks import ExtractMasks
from chess_bot.perception.create_pointclouds import CreatePointclouds

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

                builder.ExportOutput(masks.GetOutputPort('raw_prediction'),
                    f"{model_name}_raw_prediction")

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
    q0 = [0.0, 0, 0.0, -np.pi/2, 0.0, 1.60, np.pi/4]
    index = 0
    for joint_index in plant.GetJointIndices(panda):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1

    return panda


def AddWsgPanda(plant,
           panda_model_instance,
           roll=-np.pi / 4.0,
           sphere=False):
    parser = Parser(plant)
    gripper = parser.AddModelFromFile(
        FindResourceOrThrow(
            "drake/manipulation/models/"
            "wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf"))

    X_7G = RigidTransform(RollPitchYaw(np.pi / 2.0, 0, roll), [0.00, 0.00, 0.04])
    plant.WeldFrames(plant.GetFrameByName("panda_link8", panda_model_instance),
                     plant.GetFrameByName("body", gripper), X_7G)
    return gripper


def AddPandaDifferentialIK(builder, plant, frame=None):
    params = DifferentialInverseKinematicsParameters(plant.num_positions(),
                                                     plant.num_velocities())
    time_step = plant.time_step()
    q0 = plant.GetPositions(plant.CreateDefaultContext())
    params.set_nominal_joint_position(q0)
    params.set_end_effector_angular_speed_limit(2)
    params.set_end_effector_translational_velocity_limits([-2, -2, -2],
                                                          [2, 2, 2])
    # Decrease velocity to prevent oscilations in controller.
    panda_velocity_limits = 0.5 * np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])
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