import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os.path as osp

from scipy.spatial import KDTree

from pydrake.all import (
    AbstractValue,
    LeafSystem,
    RigidTransform,
    RotationMatrix
)

from chess_bot.utils.path_util import get_chessbot_src

class ICPSystem(LeafSystem):
    """
    ICP Pointcloud matching system
    """
    pieces_to_pcds = {
        'B' : 'Bishop',

        'K' : 'King',

        'N' : 'Knight',

        'P' : 'Pawn',

        'Q' : 'Queen',

        'R' : 'Rook',
    }

    def __init__(self):
        LeafSystem.__init__(self)
        self.DeclareAbstractInputPort(
            'raw_pcd_stack',
            AbstractValue.Make([np.ndarray]))

        self.DeclareAbstractOutputPort(
            'icp_pcd_stack',
            lambda: AbstractValue.Make([np.ndarray]),
            self.calc_output)

    def calc_output(self, context, output):
        pcd_stack, pieces = self.GetInputPort('raw_pcd_stack').Eval(context)
        icp_pcds = []
        for i, pcd in enumerate(pcd_stack):
            piece = pieces[i][0]  # First char is piece type
            raw_piece = pcd.xyzs()

            ref_piece_fn = osp.join(get_chessbot_src(), f'resources/reference_pcds/{ICPSystem.pieces_to_pcds[piece]}.npy')
            ref_piece = np.load(ref_piece_fn).T
            ref_piece = ref_piece[:, :raw_piece.shape[1]]  # Make model and scene the same size

            X_rawRef, _, _ = self.icp(raw_piece, ref_piece, max_iterations=30, tolerance=1e-5)
            icp_pcds.append(X_rawRef.multiply(ref_piece).T)

        output.set_value([icp_pcds, pieces])

    def icp(self, scene, model, max_iterations=20, tolerance=1e-3, upright=True):
        '''
        Perform ICP to return the correct relative transform between two set of points.
        Can throw errors if M != N.  You should randomly sample the larger into the
        same size as the smaller if possible.
        Args:
            scene: 3xN numpy array of points
            model: 3xM numpy array of points
            max_iterations: max amount of iterations the algorithm can perform.
            tolerance: tolerance before the algorithm converges.
        Returns:
        X_BA: A RigidTransform object that maps point_cloud_A on to point_cloud_B
                such that
                            X_BA.multiply(model) ~= scene,
        mean_error: Mean of all pairwise distances.
        num_iters: Number of iterations it took the ICP to converge.
        '''
        X_BA = RigidTransform()

        mean_error = 0
        num_iters = 0
        prev_error = 0

        while True:
            num_iters += 1
            # print(num_iters)

            distances, indicies = self.nearest_neighbors(scene, X_BA.multiply(model))
            # print(indicies)

            # print(distances)

            # print(indicies)
            corr_model = model[:, indicies]

            X_BA = self.least_squares_transform(scene, corr_model, upright)

            # your code here
            ##################

            mean_error = np.mean(distances)

            if abs(mean_error - prev_error) < tolerance or num_iters >= max_iterations:
                break

            prev_error = mean_error

        return X_BA, mean_error, num_iters

    def least_squares_transform(self, scene, model, upright) -> RigidTransform:
        '''
        Calculates the least-squares best-fit transform that maps corresponding
        points scene to model.
        Args:
        scene: 3xN numpy array of corresponding points
        model: 3xM numpy array of corresponding points
        Returns:
        X_BA: A RigidTransform object that maps point_cloud_A on to point_cloud_B
                such that
                            X_BA.multiply(model) ~= scene,
        '''
        # print('hi')
        p_Omc = model.T
        p_s = scene.T

        # Calculate the central points
        p_Ombar = p_Omc.mean(axis=0)
        p_sbar = p_s.mean(axis=0)

        # Calculate the "error" terms, and form the data matrix
        merr = p_Omc - p_Ombar
        serr = p_s - p_sbar
        W = np.matmul(serr.T, merr)

        # Compute R
        U, Sigma, Vt = np.linalg.svd(W)
        if upright:
            R = np.eye(3)
        else:
            R = np.matmul(U, Vt)
            if np.linalg.det(R) < 0:
                print("fixing improper rotation")
                Vt[-1, :] *= -1
                R = np.matmul(U, Vt)

        # Compute p
        p = p_sbar - np.matmul(R, p_Ombar)

        if upright:
            p[-1] = 0

        X_BA = RigidTransform(RotationMatrix(R), p)

        # X_BA = RigidTransform()
        # ##################
        # # your code here
        # ##################
        return X_BA

    def nearest_neighbors(self, scene, model):
        '''
        Find the nearest (Euclidean) neighbor in model for each
        point in scene.
        Args:
            scene: 3xN numpy array of points
            model: 3xM numpy array of points
        Returns:
            distances: (N, ) numpy array of Euclidean distances from each point in
                scene to its nearest neighbor in model.
            indices: (N, ) numpy array of the indices in model of each
                scene point's nearest neighbor - these are the c_i's
        '''
        distances = np.empty(scene.shape[1], dtype=float)
        indices = np.empty(scene.shape[1], dtype=int)

        kdtree = KDTree(model.T)
        for i in range(model.shape[1]):
            distances[i], indices[i] = kdtree.query(scene[:,i], 1, p=1)

        return distances, indices
