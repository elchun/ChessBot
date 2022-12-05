import matplotlib.pyplot as plt
import numpy as np
import os.path as osp

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import torchvision.transforms.functional as Tf

from scipy import ndimage

from pydrake.all import (
    RigidTransform,
    PointCloud,
    BaseField,
    Fields,
    AbstractValue,
    BaseField,
    LeafSystem,
    )


class CreatePointclouds(LeafSystem):
    """
    System to convert masked depth images into pointclouds.
    """
    # def __init__(self, rgbd_sensor, X_WC):
    def __init__(self, rgbd_sensor):
        LeafSystem.__init__(self)

        self.cam_info = rgbd_sensor.depth_camera_info()
        self.rgbd_sensor = rgbd_sensor

        # sensor_context = rgbd_sensor.GetMyMutableContextFromRoot(context)
        # self.X_WC = rgbd_sensor.body_pose_in_world_output_port().Eval(sensor_context)

        self.DeclareAbstractInputPort(
            'depth_image_stack',
            AbstractValue.Make([np.ndarray]))

        self.DeclareAbstractInputPort(
            'rgbd_sensor_body_pose',
            AbstractValue.Make(RigidTransform())
        )

        self.DeclareAbstractOutputPort(
            'pcd_stack',
            lambda: AbstractValue.Make([np.ndarray]),
            self.calc_output)

    def get_intrinsics(self):
        # read camera intrinsics
        cx = self.cam_info.center_x()
        cy = self.cam_info.center_y()
        fx = self.cam_info.focal_x()
        fy = self.cam_info.focal_y()
        return cx, cy, fx, fy

    def project_depth_to_pC(self, depth_pixel):
        """
        project depth pixels to points in camera frame
        using pinhole camera model
        Input:
            depth_pixels: numpy array of (nx3) or (3,)
        Output:
            pC: 3D point in camera frame, numpy array of (nx3)
        """
        # switch u,v due to python convention
        v = depth_pixel[:,0]
        u = depth_pixel[:,1]
        Z = depth_pixel[:,2]
        cx, cy, fx, fy = self.get_intrinsics()
        X = (u-cx) * Z/fx
        Y = (v-cy) * Z/fy
        pC = np.c_[X,Y,Z]
        return pC

    def get_pointcloud_np(self, depth_im, X_WC):
        u_range = np.arange(depth_im.shape[0])
        v_range = np.arange(depth_im.shape[1])
        depth_v, depth_u = np.meshgrid(v_range, u_range)
        depth_pnts = np.dstack([depth_u, depth_v, depth_im])
        depth_pnts = depth_pnts.reshape([depth_pnts.shape[0]*depth_pnts.shape[1], 3])
        pC = self.project_depth_to_pC(depth_pnts)
        p_C = pC[pC[:,2] > 0]
        p_W = X_WC.multiply(p_C.T).T
        return p_W

    def get_drake_pcd(self, pcd_np, X_WC):
        N = pcd_np.shape[0]
        pcd = PointCloud(N, Fields(BaseField.kXYZs | BaseField.kRGBs))
        pcd.mutable_xyzs()[:] = pcd_np.T  # Want (3, N) while pcd_np is (N, 3)
        pcd.EstimateNormals(radius=0.1, num_closest=30)
        pcd.FlipNormalsTowardPoint(X_WC.translation())
        pcd = pcd.VoxelizedDownSample(voxel_size=0.002)
        return pcd


    def calc_output(self, context, output):
        # sensor_context = self.rgbd_sensor.GetMyMutableContextFromRoot(context)
        # X_WC = self.rgbd_sensor.body_pose_in_world_output_port().Eval(context)

        X_WC = self.GetInputPort('rgbd_sensor_body_pose').Eval(context)
        depth_image_stack, pieces = self.GetInputPort('depth_image_stack').Eval(context)
        pcds = []
        for i in range(depth_image_stack.shape[0]):
            depth_image = depth_image_stack[i]
            pcd_np = self.get_pointcloud_np(depth_image, X_WC)
            pcd = self.get_drake_pcd(pcd_np, X_WC)
            pcds.append(pcd)
        output.set_value([pcds, pieces])

