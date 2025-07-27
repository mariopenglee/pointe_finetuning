import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import argparse
# Add these imports for safe unpickling
from torch.serialization import safe_globals
from point_e.util.point_cloud import PointCloud

# Directory containing the pointcloud .pt files
POINTCLOUD_DIR = "data/pointclouds"

# Number of pointclouds to visualize
N_SHOW = 10

def main():
    parser = argparse.ArgumentParser(description="Visualize point clouds from .pt files.")
    parser.add_argument('--dir', type=str, default=POINTCLOUD_DIR, help='Directory containing pointcloud .npz files')
    parser.add_argument('--file', type=str, default=None, help='Specific pointcloud .npz file to visualize')
    parser.add_argument('--n_show', type=int, default=N_SHOW, help='Number of pointclouds to visualize (if not specifying --file)')
    args = parser.parse_args()

    if args.file:
        pt_files = [os.path.join(args.dir, args.file)]
        if not os.path.isfile(pt_files[0]):
            raise RuntimeError(f"File {pt_files[0]} does not exist.")
    else:
        pt_files = glob(os.path.join(args.dir, "*_pc.npz"))
        if len(pt_files) == 0:
            raise RuntimeError(f"No .npz files found in {args.dir}")
        pt_files = random.sample(pt_files, min(args.n_show, len(pt_files)))

    for pt_path in pt_files:
        pc = PointCloud.load(pt_path)
        print(f"Visualizing {pt_path} as PointCloud object")
        xyz = pc.coords  # shape: [N, 3]
        # Try to get RGB channels if present
        if all(k in pc.channels for k in ('R', 'G', 'B')):
            rgb = np.stack([
                pc.channels['R'],
                pc.channels['G'],
                pc.channels['B']
            ], axis=1) / 255.0  # normalize if needed
        else:
            rgb = 'b'  # fallback to blue if no color
        fname = os.path.basename(pt_path)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=8)
        ax.set_title(fname)
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
