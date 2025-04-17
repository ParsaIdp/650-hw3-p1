# Pratik Chaudhari (pratikac@seas.upenn.edu)
# Minku Kim (minkukim@seas.upenn.edu)

import os, sys, pickle, math
from scipy import io
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from load_data import load_kitti_lidar_data, load_kitti_poses, load_kitti_calib
from utils import *

import logging
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

class map_t:
    def __init__(s, resolution=0.5):
        s.resolution = resolution
        s.xmin, s.xmax = -700, 700
        s.zmin, s.zmax = -500, 900
        # s.xmin, s.xmax = -400, 1100
        # s.zmin, s.zmax = -300, 1200

        s.szx = int(np.ceil((s.xmax - s.xmin) / s.resolution + 1))
        s.szz = int(np.ceil((s.zmax - s.zmin) / s.resolution + 1))

        # binarized map and log-odds
        s.cells = np.zeros((s.szx, s.szz), dtype=np.int8)
        s.log_odds = np.zeros(s.cells.shape, dtype=np.float64)

        # value above which we are not going to increase the log-odds,
        # and similarly we will not decrease log-odds of a cell below -max
        s.log_odds_max = 5e6
        # number of observations received for each cell
        s.num_obs_per_cell = np.zeros(s.cells.shape, dtype=np.uint64)

        # we call a cell occupied if the probability of
        # occupancy P(m_i | ... ) is >= occupied_prob_thresh
        s.occupied_prob_thresh = 0.6
        s.log_odds_thresh = np.log(s.occupied_prob_thresh / (1 - s.occupied_prob_thresh))

    def grid_cell_from_xz(s, x, z):
        """
        x and z can be 1-dimensional arrays, compute the cell indices in the map corresponding
        to these (x,y) locations. You should return an array of shape 2 x len(x). Be
        careful to handle instances when x/z go outside the map bounds, you can use
        np.clip to handle these situations.
        """
        x = np.clip(x, s.xmin, s.xmax)
        z = np.clip(z, s.zmin, s.zmax)

        x_cell = np.floor((x - s.xmin) / s.resolution).astype(np.int32)
        z_cell = np.floor((z - s.zmin) / s.resolution).astype(np.int32)

        return np.vstack((x_cell, z_cell))

class slam_t:
    """
    s is the same as self. In Python it does not really matter
    what we call self, s is shorter. As a general comment, (I believe)
    you will have fewer bugs while writing scientific code if you
    use the same/similar variable names as those in the mathematical equations.
    """
    def __init__(s, resolution=0.5, Q=1e-3*np.eye(3), resampling_threshold=0.3):
        s.lidar_log_odds_occ = np.log(9)
        s.lidar_log_odds_free = np.log(1/9.)

        # dynamics noise for the state (x, z, yaw)
        s.Q = Q

        # we resample particles if the effective number of particles
        # falls below s.resampling_threshold*num_particles
        s.resampling_threshold = resampling_threshold

        # initialize the map
        s.map = map_t(resolution)

    def read_data(s, src_dir, idx):
        """
        src_dir: location of the "data" directory
        """
        logging.info('> Reading data')
        s.idx = idx
        s.lidar_dir = src_dir + f'odometry/{s.idx}/velodyne/'
        s.poses = load_kitti_poses(src_dir + f'poses/{s.idx}.txt')
        s.lidar_files = sorted(os.listdir(src_dir + f'odometry/{s.idx}/velodyne/'))
        s.calib = load_kitti_calib(src_dir + f'calib/{s.idx}/calib.txt')

    def init_particles(s, n=100, p=None, w=None):
        """
        n: number of particles
        p: xy yaw locations of particles (3xn array)
        w: weights (array of length n)
        """
        s.n = n
        s.p = deepcopy(p) if p is not None else np.zeros((3, s.n))
        s.w = deepcopy(w) if w is not None else np.ones(n) / n

    @staticmethod
    def stratified_resampling(p, w):
        """
        Resampling step of the particle filter.
        """
        r = np.random.uniform(0, 1/(len(w)))
        c = w[0]
        i = 0
        p_new = np.zeros(p.shape)
        w_new = np.zeros(w.shape)
        for m in range(len(w)):
            u = r + m / len(w)
            while u > c:
                i += 1
                c += w[i]
            p_new[:, m] = p[:, i]
            w_new[m] = w[i]
        w_new /= np.sum(w_new)
        return p_new, w_new


    @staticmethod
    def log_sum_exp(w):
        return w.max() + np.log(np.exp(w-w.max()).sum())

    def lidar2world(s, p, points):
        """
        Transforms LiDAR points to world coordinates.

        The particle state p is now interpreted as [x, z, theta], where:
        - p[0]: x translation
        - p[1]: z translation
        - p[2]: rotation in the x-z plane

        The input 'points' is an (N, 3) array of LiDAR points in xyz.
        """

        # 1. Convert LiDAR points to homogeneous coordinates
        pts_velo = make_homogeneous_coords_3d(points.T)
    

        # 2. Transform Velodyne Frame -> Camera Frame
        Tr = s.calib
        pts_cam = Tr @ pts_velo
        

        # 3. from camera frame to world frame
        pts_2d_cam = pts_cam[[0, 2], :]
        pts_2d_cam_h = make_homogeneous_coords_2d(pts_2d_cam)
        x, z, yaw = p
        T_world_cam = get_se2(yaw, np.array([x, z]))
        pts_world = T_world_cam @ pts_2d_cam_h
    
        return pts_world[:2, :].T

    def get_control(s, t):
        """
        Use the pose at time t and t-1 to calculate what control the robot could have taken
        at time t-1 at state (x,y,th)_{t-1} to come to the current state (x,y,th)_t. We will
        assume that this is the same control that the robot will take in the function dynamics_step
        below at time t, to go to time t-1. need to use the smart_minus_2d
        function to get the difference of the two poses and we will simply
        set this to be the control.
        Extracts control in the state space [x, z, rotation] from consecutive poses.
        [x, z, theta]
        theta is the rotation around the Y-axis
              | cos  0  -sin |
        R_y = |  0   1    0  |
              |+sin  0   cos |
        R31 = +sin
        R11 =  cos
        yaw = atan2(R_31, R_11)
        """
        if t == 0:
            return np.zeros(3)
        pose_curr = s.poses[t]
        pose_prev = s.poses[t - 1]
        
        x1, z1 = pose_prev[0, 3], pose_prev[2, 3]
        x2, z2 = pose_curr[0, 3], pose_curr[2, 3]
        yaw1 = np.arctan2(pose_prev[2, 0], pose_prev[0, 0])
        yaw2 = np.arctan2(pose_curr[2, 0], pose_curr[0, 0])
        
        p1 = np.array([x1, z1, yaw1])
        p2 = np.array([x2, z2, yaw2])
        
        return smart_minus_2d(p2, p1)

    def dynamics_step(s, t):
        """
        Compute the control using get_control and perform that control on each particle to get the updated locations of the particles in the particle filter
        """
        u = s.get_control(t)
        for i in range(s.n):
            s.p[:, i] = smart_plus_2d(s.p[:, i], u)
            s.p[:, i] += np.random.multivariate_normal(np.zeros(3), s.Q)

    @staticmethod
    def update_weights(w, obs_logp):
        """
        Given the observation log-probability and the weights of particles w, calculate the
        new weights as discussed in the writeup. Make sure that the new weights are normalized
        """
        log_w = np.log(w + 1e-21) + obs_logp
        log_w -= slam_t.log_sum_exp(log_w)
        return np.exp(log_w)
    
    def observation_step(s, t):
        """
        This function does the following things
            1. updates the particles using the LiDAR observations
            2. updates map.log_odds and map.cells using occupied cells as shown by the LiDAR data
        you can also store a thresholded version of the map here for plotting later
        """

        scan = load_kitti_lidar_data(s.lidar_dir + s.lidar_files[t])
        scan = clean_point_cloud(scan)
        logps = np.zeros(s.n)

        for i in range(s.n):
            world_pts = s.lidar2world(s.p[:, i], scan[:, :3])
            grid_idxs = s.map.grid_cell_from_xz(world_pts[:, 0], world_pts[:, 1])

            valid = (grid_idxs[0] >= 0) & (grid_idxs[0] < s.map.szx) & \
                    (grid_idxs[1] >= 0) & (grid_idxs[1] < s.map.szz)

            x_idx, z_idx = grid_idxs[0][valid], grid_idxs[1][valid]
            logps[i] = np.sum(s.map.cells[x_idx, z_idx] == 1)

        s.w = s.update_weights(s.w, logps)

        # Select best particle to update the map
        best_idx = np.argmax(s.w)
        best_pose = s.p[:, best_idx]
        best_pts = s.lidar2world(best_pose, scan[:, :3])
        grid_idxs = s.map.grid_cell_from_xz(best_pts[:, 0], best_pts[:, 1])

        valid = (grid_idxs[0] >= 0) & (grid_idxs[0] < s.map.szx) & \
                (grid_idxs[1] >= 0) & (grid_idxs[1] < s.map.szz)

        x_idx, z_idx = grid_idxs[0][valid], grid_idxs[1][valid]

        # Update map
        s.map.log_odds[x_idx, z_idx] += s.lidar_log_odds_occ
        s.map.log_odds = np.clip(s.map.log_odds, -s.map.log_odds_max, s.map.log_odds_max)

        s.map.cells = (s.map.log_odds >= s.map.log_odds_thresh).astype(np.int8)

    def resample_particles(s):
        """
        Resampling is a (necessary) but problematic step which introduces a lot of variance
        in the particles. We should resample only if the effective number of particles
        falls below a certain threshold (resampling_threshold). A good heuristic to
        calculate the effective particles is 1/(sum_i w_i^2) where w_i are the weights
        of the particles, if this number of close to n, then all particles have about
        equal weights and we do not need to resample
        """
        e = 1/np.sum(s.w**2)
        logging.debug('> Effective number of particles: {}'.format(e))
        if e/s.n < s.resampling_threshold:
            s.p, s.w = s.stratified_resampling(s.p, s.w)
            logging.debug('> Resampling')
