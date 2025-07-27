import os
import argparse
import torch
from point_e.util.point_cloud import PointCloud
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.pc_to_mesh import marching_cubes_mesh


def pointcloud_to_mesh(pc_path, mesh_path, device):
    # Load point cloud
    pc = PointCloud.load(pc_path)

    # Load SDF model
    name = 'sdf'
    model = model_from_config(MODEL_CONFIGS[name], device)
    model.eval()
    model.load_state_dict(load_checkpoint(name, device))

    # Generate mesh
    mesh = marching_cubes_mesh(
        pc=pc,
        model=model,
        batch_size=4096,
        grid_size=32,  # You can increase for higher resolution
        progress=True,
    )

    # Save mesh as PLY
    with open(mesh_path, 'wb') as f:
        mesh.write_ply(f)
    print(f"Mesh saved to {mesh_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert point clouds to meshes using Point-E SDF model.")
    parser.add_argument('--input', type=str, required=True, help='Input .npz point cloud or directory of .npz files')
    parser.add_argument('--output', type=str, required=False, help='Output mesh file or directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')
    args = parser.parse_args()

    device = torch.device(args.device)

    if os.path.isdir(args.input):
        out_dir = args.output or args.input
        os.makedirs(out_dir, exist_ok=True)
        for fname in os.listdir(args.input):
            if fname.endswith('.npz'):
                in_fp = os.path.join(args.input, fname)
                out_fp = os.path.join(out_dir, os.path.splitext(fname)[0] + '.ply')
                pointcloud_to_mesh(in_fp, out_fp, device)
    else:
        out_fp = args.output or os.path.splitext(args.input)[0] + '.ply'
        pointcloud_to_mesh(args.input, out_fp, device)

if __name__ == '__main__':
    main()
