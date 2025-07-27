import os
from glob import glob
import trimesh
import numpy as np
import PIL.Image
from point_e.util.point_cloud import PointCloud

def sample_point_cloud(mesh: trimesh.Trimesh, num_points: int = 1024):
    """
    Samples a point cloud from a mesh surface *and* normalizes it
    into a unit sphere centered at the origin—just like Point‑E does.
    """
    # 1) Uniformly sample points on the mesh
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    
    # 2) Center & scale to unit sphere
    centroid = points.mean(axis=0)
    points  -= centroid
    scale    = np.max(np.linalg.norm(points, axis=1))
    points  /= scale

    return points

def sample_point_cloud_with_color(mesh: trimesh.Trimesh, num_points: int = 1024):
    """
    Samples a point cloud from a mesh surface *and* normalizes it
    into a unit sphere centered at the origin—just like Point‑E does.
    Also samples color at each point if available, including from textures.
    Returns (points, colors) where colors is None if not available.
    """
    # 1) Uniformly sample points on the mesh
    sample_result = trimesh.sample.sample_surface(mesh, num_points)
    if len(sample_result) == 3:
        points, face_indices, barycentric = sample_result
    else:
        points, face_indices = sample_result
        barycentric = None
    
    # 2) Center & scale to unit sphere
    centroid = points.mean(axis=0)
    points  -= centroid
    scale    = np.max(np.linalg.norm(points, axis=1))
    points  /= scale

    # 3) Try to get color for each sampled point
    colors = None
    visual = getattr(mesh, 'visual', None)
    # Try face or vertex colors first
    if visual is not None and hasattr(visual, 'face_colors') and visual.face_colors is not None:
        face_colors = visual.face_colors[face_indices, :3]
        colors = face_colors.astype(np.float32) / 255.0
    elif visual is not None and hasattr(visual, 'vertex_colors') and visual.vertex_colors is not None:
        faces = mesh.faces[face_indices]
        vertex_colors = visual.vertex_colors[:, :3]
        colors = vertex_colors[faces].mean(axis=1).astype(np.float32) / 255.0
    # Try texture sampling if available
    elif visual is not None:
        uv = getattr(visual, 'uv', None)
        material = getattr(visual, 'material', None)
        image = getattr(material, 'image', None) if material is not None else None
        if uv is not None and image is not None:
            faces = mesh.faces[face_indices]
            if barycentric is None:
                uv_coords = uv[faces[:, 0]]
            else:
                uv_coords = (uv[faces] * barycentric[:, :, None]).sum(axis=1)
            if not isinstance(image, np.ndarray):
                image = np.array(image)
            h, w = image.shape[:2]
            px = np.clip((uv_coords[:, 0] * w).astype(int), 0, w - 1)
            py = np.clip(((1 - uv_coords[:, 1]) * h).astype(int), 0, h - 1)
            colors = image[py, px, :3].astype(np.float32) / 255.0
    return points, colors

if __name__ == "__main__":
    input_root = "data/objects"
    output_dir = "data/pointclouds"
    os.makedirs(output_dir, exist_ok=True)

    for obj_path in glob(os.path.join(input_root, "*", "*.obj")):
        mesh = trimesh.load(obj_path, process=True)
        # Ensure mesh is a Trimesh object (not a Scene)
        if isinstance(mesh, trimesh.Scene):
            if hasattr(mesh, 'geometry') and len(mesh.geometry) > 0:
                mesh = list(mesh.geometry.values())[0]
            else:
                print(f"Skipping {obj_path}: Scene contains no geometry.")
                continue
        if not isinstance(mesh, trimesh.Trimesh):
            print(f"Skipping {obj_path}: not a Trimesh.")
            continue
        stem = os.path.splitext(os.path.basename(obj_path))[0]
        print(f"Sampling {stem} → 1024 points, normalized")

        points, colors = sample_point_cloud_with_color(mesh, num_points=1024)

        if colors is None:
            print(f"Warning: {obj_path} has no color information, padding with zeros.")
            colors = np.zeros_like(points)
        # Save as PointCloud npz
        channels = {'R': (colors[:,0]*255).astype(np.uint8),
                    'G': (colors[:,1]*255).astype(np.uint8),
                    'B': (colors[:,2]*255).astype(np.uint8)}
        pc_obj = PointCloud(coords=points, channels=channels)
        out_fp = os.path.join(output_dir, f"{stem}_pc.npz")
        pc_obj.save(out_fp)
        print(f"Saved → {out_fp} (coords shape: {points.shape})")

    print("\nAll done! Your Point‑E–style point clouds are in 'pointclouds/'")
