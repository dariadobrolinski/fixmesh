import trimesh
import numpy as np
import open3d as o3d
import pymesh
from scipy.spatial import cKDTree

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def detach_repair(
    input,
    contact_threshold=0.06,
    max_iters=10,
    num_sample_points=50000
):
    """
    Fix self-intersections by detaching mesh.

    Args:
        input (PyMesh): The input mesh.
        contact_thresold (float, optional): The separation distance between meshes. Defaults to 0.06.
        max_iters (int, optional): The number of iterations to repair. Defaults to 10.
        num_sample_points (int, optional): The number of sampling points of the "reference" mesh.

    Returns:
        trimesh.Mesh:
            A new mesh with no self-intersections.

    1) Load a mesh and split into submeshes.
    2) Identify smallest-volume piece as 'inner', largest-volume piece as 'outer'.
    3) Convert each to Open3D. 
    4) Iteratively shrink the inner mesh so it stays inside the outer.

    Adjust the values of max_iters and num_sample_points based on your input mesh.
    
    Increasing contact_threshold increases robustness. 

    """
    tm = input
    submeshes = tm.split(only_watertight=True)
    
    if len(submeshes) < 2:
        raise ValueError("Expected at least two submeshes (inner and outer).")

    # Sort by volume: smallest -> "inner", largest -> "outer"
    submeshes_sorted = sorted(submeshes, key=lambda m: abs(m.volume))
    mesh_inner_tm = submeshes_sorted[0]
    mesh_outer_tm = submeshes_sorted[-1]

    # Convert to Open3D
    mesh_inner_o3d = _trimesh_to_open3d(mesh_inner_tm)
    mesh_outer_o3d = _trimesh_to_open3d(mesh_outer_tm)

    # Compute normals for better sampling
    print("Compute normals for better sampling")
    mesh_inner_o3d.compute_vertex_normals()
    mesh_outer_o3d.compute_vertex_normals()

    # Shrink the inner mesh
    mesh_inner_o3d = _shrink_inner_mesh(
        inner_o3d_mesh=mesh_inner_o3d,
        outer_o3d_mesh=mesh_outer_o3d,
        contact_threshold=contact_threshold,
        max_iters=max_iters,
        num_sample_points=num_sample_points,
    )

    combined = mesh_inner_o3d + mesh_outer_o3d
    return combined


def _shrink_inner_mesh(
    inner_o3d_mesh: o3d.geometry.TriangleMesh,
    outer_o3d_mesh: o3d.geometry.TriangleMesh,
    contact_threshold=0.06,
    max_iters=10,
    num_sample_points=50000
):
    """
    Same movement logic as original, but uses:
    - cKDTree for nearest surface distance
    - Open3D RaycastingScene for fast inside/outside test
    """

    # Sample outer mesh surface and build KD-tree
    print("Sample outer mesh and build KD-tree")
    pcd_outer = outer_o3d_mesh.sample_points_poisson_disk(number_of_points=num_sample_points)
    outer_pts = np.asarray(pcd_outer.points)
    kd_tree = cKDTree(outer_pts)

    # Prepare inner vertices and center
    print("Prepare inner vertices and center")
    inner_verts = np.asarray(inner_o3d_mesh.vertices)
    center_inner = np.mean(inner_verts, axis=0)

    # Set up Open3D RaycastingScene for fast inside/outside test
    print("Set up RaycastingScene")
    outer_mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(outer_o3d_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(outer_mesh_tensor)

    for _iter in range(max_iters):
        print(f"{_iter} iteration")

        changed_any = False

        # Get signed distance from outer surface (Open3D Tensor input)
        sdf = scene.compute_signed_distance(
            o3d.core.Tensor(inner_verts, dtype=o3d.core.float32)
        ).numpy()
        inside_mask = sdf < 0  # same logic as trimesh.contains

        # Nearest surface distance (vectorized)
        nn_dists, _ = kd_tree.query(inner_verts, k=1)

        # Direction vectors from center
        directions = inner_verts - center_inner
        lengths = np.linalg.norm(directions, axis=1)
        directions_unit = np.divide(
            directions,
            lengths[:, np.newaxis],
            out=np.zeros_like(directions),
            where=lengths[:, np.newaxis] > 1e-12
        )

        # Determine which vertices to move
        move_mask = ~inside_mask | (nn_dists < contact_threshold)
        offsets = np.zeros_like(nn_dists)

        # Case 1: outside → pull in
        offsets[~inside_mask] = nn_dists[~inside_mask] + contact_threshold

        # Case 2: inside but too close → nudge inward
        close_inside = inside_mask & (nn_dists < contact_threshold)
        offsets[close_inside] = contact_threshold - nn_dists[close_inside]

        # Apply movement
        if np.any(offsets > 0):
            inner_verts[move_mask] -= directions_unit[move_mask] * offsets[move_mask, np.newaxis]
            changed_any = True

        if not changed_any:
            print("Converged")
            break

    # Update Open3D mesh
    inner_o3d_mesh.vertices = o3d.utility.Vector3dVector(inner_verts)
    return inner_o3d_mesh


def _trimesh_to_open3d(tri_mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(tri_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(tri_mesh.faces)
    return o3d_mesh

def _open3d_to_trimesh(o3d_mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)



#####
def _detach_repair_v1(
    input_path,
    contact_threshold=0.06,
    max_iters=10,
    num_sample_points=50000
):
    """
    Fix self-intersections by detaching mesh.

    Args:
        input_path (string): The path to the input mesh.
        contact_thresold (float, optional): The separation distance between meshes. Defaults to 0.06.
        max_iters (int, optional): The number of iterations to repair. Defaults to 10.
        num_sample_points (int, optional): The number of sampling points of the "reference" mesh.

    Returns:
        trimesh.Mesh:
            A new mesh with no self-intersections.

    1) Load a mesh and split into submeshes.
    2) Identify smallest-volume piece as 'inner', largest-volume piece as 'outer'.
    3) Convert each to Open3D. 
    4) Iteratively shrink the inner mesh so it stays inside the outer.


    
    """
    tm = trimesh.load(input_path, process=False)
    submeshes = tm.split(only_watertight=True)
    
    # Sort by volume: smallesr -> "inner", larger -> "outer"
    submeshes_sorted = sorted(submeshes, key=lambda m: m.volume)
    mesh_inner_tm = submeshes_sorted[0]
    mesh_outer_tm = submeshes_sorted[-1]

    # Convert to Open3D
    mesh_inner_o3d = _trimesh_to_open3d(mesh_inner_tm)
    mesh_outer_o3d = _trimesh_to_open3d(mesh_outer_tm)

    # Compute normals for better sampling
    mesh_inner_o3d.compute_vertex_normals()
    mesh_outer_o3d.compute_vertex_normals()

    # Shrink the inner mesh
    mesh_inner_o3d = _shrink_inner_mesh_v1(
        inner_o3d_mesh=mesh_inner_o3d,
        outer_o3d_mesh=mesh_outer_o3d,
        contact_threshold=contact_threshold,
        max_iters=max_iters,
        num_sample_points=num_sample_points
    )

    combined = mesh_inner_o3d + mesh_outer_o3d
    return combined

def _detach_repair_v2(
    input,
    contact_threshold=0.06,
    max_iters=10,
    num_sample_points=50000
):
    """
    Fix self-intersections by detaching mesh.

    Args:
        input (mesh): The input mesh.
        contact_thresold (float, optional): The separation distance between meshes. Defaults to 0.06.
        max_iters (int, optional): The number of iterations to repair. Defaults to 10.
        num_sample_points (int, optional): The number of sampling points of the "reference" mesh.

    Returns:
        trimesh.Mesh:
            A new mesh with no self-intersections.

    1) Load a mesh and split into submeshes.
    2) Identify smallest-volume piece as 'inner', largest-volume piece as 'outer'.
    3) Convert each to Open3D. 
    4) Iteratively shrink the inner mesh so it stays inside the outer.


    
    """
    tm = input
    submeshes = tm.split(only_watertight=True)
    
    # Sort by volume: smallesr -> "inner", larger -> "outer"
    submeshes_sorted = sorted(submeshes, key=lambda m: m.volume)
    mesh_inner_tm = submeshes_sorted[0]
    mesh_outer_tm = submeshes_sorted[-1]

    # Convert to Open3D
    mesh_inner_o3d = _trimesh_to_open3d(mesh_inner_tm)
    mesh_outer_o3d = _trimesh_to_open3d(mesh_outer_tm)

    # Compute normals for better sampling
    mesh_inner_o3d.compute_vertex_normals()
    mesh_outer_o3d.compute_vertex_normals()

    # Shrink the inner mesh
    mesh_inner_o3d = _shrink_inner_mesh_v1(
        inner_o3d_mesh=mesh_inner_o3d,
        outer_o3d_mesh=mesh_outer_o3d,
        contact_threshold=contact_threshold,
        max_iters=max_iters,
        num_sample_points=num_sample_points
    )

    combined = mesh_inner_o3d + mesh_outer_o3d
    return combined

def _shrink_inner_mesh_v1(
    inner_o3d_mesh: o3d.geometry.TriangleMesh,
    outer_o3d_mesh: o3d.geometry.TriangleMesh,
    contact_threshold=0.06,
    max_iters=10,
    num_sample_points=50000
):
    """
    Iteratively shrinks inner mesh so it lies inside outer mesh

    1) sample the outer mesh.
    2) see which points are inside or outside.
    3) For each iteration:
       a) Find which vertices of inner mesh are outside -> pull them inward.
       b) For vertices that are inside but too close to the outer surface,
          also pull them inward slightly.
    """

    # Build a point cloud
    pcd_outer = outer_o3d_mesh.sample_points_poisson_disk(number_of_points=num_sample_points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_outer)
    
    # Access the inner mesh vertices
    outer_tri = _open3d_to_trimesh(outer_o3d_mesh)
    inner_verts = np.asarray(inner_o3d_mesh.vertices)
    center_inner = np.mean(inner_verts, axis=0)

    for _iter in range(max_iters):
        changed_any = False

        # Check which vertices are inside vs. outside
        inside_mask = outer_tri.contains(inner_verts)
        
        # Iterate over each vertex
        for i in range(len(inner_verts)):
            v = inner_verts[i]

            # Perform KD-tree search once
            k, idx, dist_sq = pcd_tree.search_knn_vector_3d(v, 1)
            if k == 0:
                continue  # No nearest point found, skip

            dist = np.sqrt(dist_sq[0])  # Distance to closest surface

            direction = v - center_inner
            length = np.linalg.norm(direction)
            if length > 1e-12:
                direction_unit = direction / length

            if not inside_mask[i]:
                offset = dist + contact_threshold
                if offset > 0:
                    inner_verts[i] = v - offset * direction_unit
                    changed_any = True
                continue

            # If inside, check distance to nearest surface point
            if dist < contact_threshold:
                # Pull inward so that it is at least contact_threshold from the outer surface
                offset = contact_threshold - dist
                if offset > 0:
                    inner_verts[i] = v - offset * direction_unit
                    changed_any = True

        if not changed_any:
            # No vertex changed => we've converged
            break

    inner_o3d_mesh.vertices = o3d.utility.Vector3dVector(inner_verts)
    return inner_o3d_mesh

def detach_repair_raw_2(
    input,
    contact_threshold=0.06,
    max_iters=10,
    num_sample_points=50000
):
    """
    Fix self-intersections by detaching mesh.

    Args:
        input (mesh): The input mesh.
        contact_threshold (float, optional): The separation distance between meshes. Defaults to 0.06.
        max_iters (int, optional): The number of iterations to repair. Defaults to 10.
        num_sample_points (int, optional): The number of sampling points of the "reference" mesh.

    Returns:
        o3d.geometry.TriangleMesh: A new mesh with no self-intersections.
    """
    tm = input
    submeshes = tm.split(only_watertight=True)
    
    if len(submeshes) < 2:
        raise ValueError("Expected at least two submeshes (inner and outer).")

    # Sort by volume: smallest -> "inner", largest -> "outer"
    submeshes_sorted = sorted(submeshes, key=lambda m: abs(m.volume))
    mesh_inner_tm = submeshes_sorted[0]
    mesh_outer_tm = submeshes_sorted[-1]

    # Convert to Open3D
    mesh_inner_o3d = _trimesh_to_open3d(mesh_inner_tm)
    mesh_outer_o3d = _trimesh_to_open3d(mesh_outer_tm)

    # Compute normals for better sampling
    print("Compute normals for better sampling")
    mesh_inner_o3d.compute_vertex_normals()
    mesh_outer_o3d.compute_vertex_normals()

    # Shrink the inner mesh
    mesh_inner_o3d = _shrink_inner_mesh_3(
        inner_o3d_mesh=mesh_inner_o3d,
        outer_o3d_mesh=mesh_outer_o3d,
        contact_threshold=contact_threshold,
        max_iters=max_iters,
        num_sample_points=num_sample_points,
    )

    combined = mesh_inner_o3d + mesh_outer_o3d
    return combined

def _shrink_inner_mesh_2(
    inner_o3d_mesh: o3d.geometry.TriangleMesh,
    outer_o3d_mesh: o3d.geometry.TriangleMesh,
    contact_threshold=0.06,
    max_iters=10,
    num_sample_points=50000
):
    """
    Logic-preserving optimization of original shrink function using cKDTree
    and vectorized offset application.
    """

    # Sample outer surface points + build fast KD-tree
    print("Sample outer mesh and build KD-tree")
    pcd_outer = outer_o3d_mesh.sample_points_poisson_disk(number_of_points=num_sample_points)
    outer_pts = np.asarray(pcd_outer.points)
    kd_tree = cKDTree(outer_pts)

    # Get inner mesh vertices and center
    print("Prepare inner vertices and center")
    inner_verts = np.asarray(inner_o3d_mesh.vertices)
    center_inner = np.mean(inner_verts, axis=0)

    # Prepare inside checker
    outer_tri = _open3d_to_trimesh(outer_o3d_mesh)

    for _iter in range(max_iters):
        print(f"{_iter} iterations")
        changed_any = False

        # Inside/outside test
        inside_mask = outer_tri.contains(inner_verts)

        # Nearest surface distance (vectorized)
        nn_dists, _ = kd_tree.query(inner_verts, k=1)

        # Compute direction vectors (centered)
        directions = inner_verts - center_inner
        lengths = np.linalg.norm(directions, axis=1)
        directions_unit = np.divide(
            directions,
            lengths[:, np.newaxis],
            out=np.zeros_like(directions),
            where=lengths[:, np.newaxis] > 1e-12
        )

        # Build move mask + offsets
        move_mask = ~inside_mask | (nn_dists < contact_threshold)
        offsets = np.zeros_like(nn_dists)

        # Case 1: vertex outside → pull in (dist + threshold)
        outside = ~inside_mask
        offsets[outside] = nn_dists[outside] + contact_threshold

        # Case 2: vertex inside but too close → pull slightly
        close_inside = inside_mask & (nn_dists < contact_threshold)
        offsets[close_inside] = contact_threshold - nn_dists[close_inside]

        # Apply movement
        if np.any(offsets > 0):
            inner_verts[move_mask] -= directions_unit[move_mask] * offsets[move_mask, np.newaxis]
            changed_any = True

        if not changed_any:
            break

    # Update mesh
    inner_o3d_mesh.vertices = o3d.utility.Vector3dVector(inner_verts)
    return inner_o3d_mesh

def _shrink_inner_mesh_3(
    inner_o3d_mesh: o3d.geometry.TriangleMesh,
    outer_o3d_mesh: o3d.geometry.TriangleMesh,
    contact_threshold=0.06,
    max_iters=10,
    num_sample_points=50000
):
    """
    Same movement logic as original, but uses:
    - cKDTree for nearest surface distance
    - Open3D RaycastingScene for fast inside/outside test
    """

    # Sample outer mesh surface and build KD-tree
    print("Sample outer mesh and build KD-tree")
    pcd_outer = outer_o3d_mesh.sample_points_poisson_disk(number_of_points=num_sample_points)
    outer_pts = np.asarray(pcd_outer.points)
    kd_tree = cKDTree(outer_pts)

    # Prepare inner vertices and center
    print("Prepare inner vertices and center")
    inner_verts = np.asarray(inner_o3d_mesh.vertices)
    center_inner = np.mean(inner_verts, axis=0)

    # Set up Open3D RaycastingScene for fast inside/outside test
    print("Set up RaycastingScene")
    outer_mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(outer_o3d_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(outer_mesh_tensor)

    for _iter in range(max_iters):
        print(f"{_iter} iteration")

        changed_any = False

        # Get signed distance from outer surface (Open3D Tensor input)
        sdf = scene.compute_signed_distance(
            o3d.core.Tensor(inner_verts, dtype=o3d.core.float32)
        ).numpy()
        inside_mask = sdf < 0  # same logic as trimesh.contains

        # Nearest surface distance (vectorized)
        nn_dists, _ = kd_tree.query(inner_verts, k=1)

        # Direction vectors from center
        directions = inner_verts - center_inner
        lengths = np.linalg.norm(directions, axis=1)
        directions_unit = np.divide(
            directions,
            lengths[:, np.newaxis],
            out=np.zeros_like(directions),
            where=lengths[:, np.newaxis] > 1e-12
        )

        # Determine which vertices to move
        move_mask = ~inside_mask | (nn_dists < contact_threshold)
        offsets = np.zeros_like(nn_dists)

        # Case 1: outside → pull in
        offsets[~inside_mask] = nn_dists[~inside_mask] + contact_threshold

        # Case 2: inside but too close → nudge inward
        close_inside = inside_mask & (nn_dists < contact_threshold)
        offsets[close_inside] = contact_threshold - nn_dists[close_inside]

        # Apply movement
        if np.any(offsets > 0):
            inner_verts[move_mask] -= directions_unit[move_mask] * offsets[move_mask, np.newaxis]
            changed_any = True

        if not changed_any:
            print("Converged")
            break

    # Update Open3D mesh
    inner_o3d_mesh.vertices = o3d.utility.Vector3dVector(inner_verts)
    return inner_o3d_mesh




def _trimesh_to_open3d(tri_mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(tri_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(tri_mesh.faces)
    return o3d_mesh

def _open3d_to_trimesh(o3d_mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
