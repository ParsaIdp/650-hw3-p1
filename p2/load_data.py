# Pratik Chaudhari (pratikac@seas.upenn.edu)
# Minku Kim (minkukim@seas.upenn.edu)

import numpy as np
import os
import struct
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import clean_point_cloud

def load_kitti_lidar_data(bin_path):
    """
    Load Velodyne LiDAR data from KITTI dataset.
    Each .bin file contains 3D point cloud data in the format: [x, y, z, intensity]
    """
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points

def load_kitti_poses(poses_file):
    """
    Load ground truth poses from KITTI dataset.
    """
    poses = []
    with open(poses_file, 'r') as f:
        for line in f.readlines():
            T = np.array([float(x) for x in line.strip().split()]).reshape(3, 4)
            poses.append(T)
    return np.array(poses)

def load_kitti_calib(calib_file):
    """
    Load calibration from KITTI dataset.
    """
    calib = {}
    with open(calib_file, 'r') as f:
        for line in f:
            if line.strip():
                key, value = line.split(":", 1)
                calib[key.strip()] = np.array([float(x) for x in value.split()])
    Tr = calib['Tr'].reshape(3, 4)
    return Tr

def show_kitti_lidar(point_cloud):
    """
    Visualize a single KITTI LiDAR point cloud (Top-Down View).
    """
    point_cloud = clean_point_cloud(point_cloud)

    plt.figure(figsize=(10, 10))
    plt.scatter(point_cloud[:, 0], point_cloud[:, 1], s=0.5, c=point_cloud[:, 3], cmap='jet')
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title("KITTI LiDAR Point Cloud (Top-Down View)")
    plt.colorbar(label="Intensity")
    plt.axis("equal")
    plt.show() 

def show_kitti_lidar_sequence(directory, start_idx=0, end_idx=10, interval=100):
    """
    Visualize and animate KITTI LiDAR point cloud sequence.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    first_file = os.path.join(directory, f"{start_idx:06d}.bin")
    first_cloud = load_kitti_lidar_data(first_file)
    first_cloud = clean_point_cloud(first_cloud)
    
    scatter = ax.scatter(first_cloud[:, 0], first_cloud[:, 1], s=0.5, c=first_cloud[:, 3], cmap='jet')

    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Z (meters)")
    ax.set_title("KITTI LiDAR Point Cloud (Top-Down View)")
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.axis("equal")
    cbar = plt.colorbar(scatter, ax=ax, label="Intensity")

    bin_files = [os.path.join(directory, f"{i:06d}.bin") for i in range(start_idx, end_idx + 1)]

    def update(frame):
        bin_path = bin_files[frame]
        point_cloud = load_kitti_lidar_data(bin_path)
        point_cloud = clean_point_cloud(point_cloud)

        scatter.set_offsets(point_cloud[:, :2])
        scatter.set_array(point_cloud[:, 3])
        ax.set_title(f"Frame: {frame + start_idx:06d}")
        return scatter,

    ani = animation.FuncAnimation(fig, update, frames=len(bin_files), interval=interval, blit=False)
    plt.show()

def trajectory3d(poses_dir):
    poses = load_kitti_poses(poses_dir)
    fig = plt.figure(figsize=(7,6))
    traj = fig.add_subplot(111, projection='3d')
    traj.plot(poses[:,:,3][:,0], poses[:,:,3][:,1], poses[:,:,3][:,2])
    traj.set_xlabel('x')
    traj.set_ylabel('y')
    traj.set_zlabel('z')
    plt.show()

def trajectory2d(poses_dir):
    poses = load_kitti_poses(poses_dir)
    x = poses[:, 0, 3]
    z = poses[:, 2, 3]
    plt.figure(figsize=(8, 8))
    plt.plot(x, z, linestyle='-', label='Trajectory')
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.title('Top View of Pose Trajectory')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

if __name__=='__main__':
    # Visualize a single top-view Lidar 
    lidar_file = "./KITTI/odometry/01/velodyne/" + "000000.bin"
    lidar_data = load_kitti_lidar_data(lidar_file)
    show_kitti_lidar(lidar_data)

    # Visualize a sequence of top-view Lidar
    lidar_dir = "./KITTI/odometry/01/velodyne/"
    show_kitti_lidar_sequence(lidar_dir, start_idx=0, end_idx=4660, interval=1)

    # Load poses
    # pose_file = "./KITTI/poses/02.txt"
    # poses = load_kitti_poses(pose_file)
    # print(poses.shape)

    # Visualize Pose Trajectory
    pose_file = "./KITTI/poses/01.txt"
    trajectory2d(pose_file)

