import trimesh
import numpy as np
import pymesh
import os

def voxelize_and_save(mesh_path, output_dir, voxel_size=0.01):
    # Load the input mesh
    mesh = trimesh.load(mesh_path)
    components = mesh.split(only_watertight=True)

    # Assign labels
    labels = list(range(1, len(components) + 1))

    # Compute global bounds across all components
    global_origin = mesh.bounds[0]
    global_max = mesh.bounds[1]
    extent = global_max - global_origin
    grid_shape = np.ceil(extent / voxel_size).astype(int)

    # Allocate empty label array
    label_array = np.zeros(grid_shape, dtype=np.uint16)

    for i, comp in enumerate(components):
        label = labels[i]
        vgrid = comp.voxelized(voxel_size)

        # Find lower corner of voxel grid
        local_origin = vgrid.points.min(axis=0) - (voxel_size / 2)

        # Offset from global origin
        offset = ((local_origin - global_origin) / voxel_size).astype(int)

        for voxel in vgrid.sparse_indices:
            global_idx = tuple(voxel + offset)
            if all(0 <= gi < s for gi, s in zip(global_idx, grid_shape)):
                label_array[global_idx] = label

    os.makedirs(output_dir, exist_ok=True)

    # Save label array
    raw_path = os.path.join(output_dir, "voxel_labels.raw")
    meta_path = os.path.join(output_dir, "voxel_labels_meta.txt")
    label_array.transpose(2, 1, 0).ravel(order="C").astype(np.uint16).tofile(raw_path)

    # Save metadata
    with open(meta_path, "w") as f:
        f.write(f"{label_array.shape[0]} {label_array.shape[1]} {label_array.shape[2]}\n")
        f.write(f"{voxel_size} {voxel_size} {voxel_size}\n")
        f.write(f"{global_origin[0]} {global_origin[1]} {global_origin[2]}\n")

    return label_array

def compute_median_edge_length(mesh):
    """
    Compute the median edge length of a mesh. 
    """
    face_indices = mesh.faces
    verts = mesh.vertices
    
    edge_lengths = []
    for f in face_indices:
        # Each face has 3 edges: (f[0], f[1]), (f[1], f[2]), (f[2], f[0])
        e1 = np.linalg.norm(verts[f[0]] - verts[f[1]])
        e2 = np.linalg.norm(verts[f[1]] - verts[f[2]])
        e3 = np.linalg.norm(verts[f[2]] - verts[f[0]])
        edge_lengths.extend([e1, e2, e3])
    
    return np.median(edge_lengths)


if __name__ == "__main__":
    mesh_files = [
        "../../examples/archive/data/two_spheres.ply",
        "../../examples/archive/data/three_spheres.ply",
        "../../examples/archive/data/sphere_cube.ply"
    ]

    output_dirs = [
        "../../examples/archive/SNResult/case1",
        "../../examples/archive/SNResult/case2",
        "../../examples/archive/SNResult/case3"
    ]

    for mesh_file, output_dir in zip(mesh_files, output_dirs):
        pmesh = pymesh.load_mesh(mesh_file)
        mesh = trimesh.load(mesh_file)

        median_edge_length = compute_median_edge_length(pmesh)
        voxel_size = median_edge_length

        voxelize_and_save(mesh_file, output_dir, voxel_size=voxel_size/4)

