# Based on Exercise 4.6 of https://manipulation.csail.mit.edu/

import numpy as np
from pydrake.all import (PointCloud, Rgba, RigidTransform, RotationMatrix,
                         StartMeshcat)
from scipy.spatial import KDTree


def icp(scene, model, max_iterations=20, tolerance=1e-3, upright=True):
    '''
    Perform ICP to return the correct relative transform between two set of points.
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

        distances, indicies = nearest_neighbors(scene, X_BA.multiply(model))
        # print(indicies)

        # print(distances)

        # print(indicies)
        corr_model = model[:, indicies]

        X_BA = least_squares_transform(scene, corr_model, upright)

        # your code here
        ##################

        mean_error = np.mean(distances)

        if abs(mean_error - prev_error) < tolerance or num_iters >= max_iterations:
            break

        prev_error = mean_error

    return X_BA, mean_error, num_iters

def least_squares_transform(scene, model, upright) -> RigidTransform:
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

def nearest_neighbors(scene, model):
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


