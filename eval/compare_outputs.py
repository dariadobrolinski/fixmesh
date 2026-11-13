"""
Compare repaired meshes against the original input.

For each output mesh this reports basic health (faces, components,
watertight, remaining self-intersecting pairs) plus how far the original
surface moved. "Lost" vertices are original vertices with no output vertex
within one median edge length -- a rough measure of how much cortex the
repair cut away or flattened. "% deep lost" is the same, but only over the
deepest 25% of original vertices (distance to their hemisphere's convex
hull), i.e. the sulci.

To run:
python eval/compare_outputs.py <original.stl> <output1.stl> [<output2.stl> ...]
"""

import argparse
import os

import numpy as np
import pymesh
import trimesh
from scipy.spatial import cKDTree


def _median_edge_length(vertices, faces):
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    return float(np.median(np.linalg.norm(
        vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)))


def _hull_depth(vertices, faces):
    """Distance from each vertex to the convex hull of its own component."""
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    depth = np.zeros(len(vertices))
    for component in trimesh.graph.connected_components(mesh.edges):
        hull = trimesh.convex.convex_hull(vertices[component])
        squared, _, _ = pymesh.distance_to_mesh(
            pymesh.form_mesh(hull.vertices, hull.faces), vertices[component])
        depth[component] = np.sqrt(squared)
    return depth


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("original")
    parser.add_argument("outputs", nargs="+")
    args = parser.parse_args()

    original = pymesh.load_mesh(args.original)
    original, _ = pymesh.remove_duplicated_vertices(original)
    edge = _median_edge_length(original.vertices, original.faces)
    depth = _hull_depth(original.vertices, original.faces)
    deep = depth >= np.percentile(depth, 75)
    print(f"original: {original.num_faces:,} faces, median edge {edge:.4f}, "
          f"deep = more than {depth[deep].min():.2f} from the hull\n")

    header = ("mesh", "faces", "comps", "watertight", "crossings",
              "mean dev", "p99 dev", "max dev", "% lost", "% deep lost")
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for path in args.outputs:
        mesh = pymesh.load_mesh(path)
        mesh, _ = pymesh.remove_duplicated_vertices(mesh)
        crossings = len(pymesh.detect_self_intersection(mesh))
        watertight = mesh.is_closed() and mesh.is_manifold()
        dist, _ = cKDTree(mesh.vertices).query(original.vertices)
        lost = 100.0 * np.mean(dist > edge)
        deep_lost = 100.0 * np.mean(dist[deep] > edge)
        row = (os.path.basename(path), f"{mesh.num_faces:,}",
               str(mesh.num_surface_components), str(watertight),
               f"{crossings:,}", f"{dist.mean():.4f}",
               f"{np.percentile(dist, 99):.4f}", f"{dist.max():.4f}",
               f"{lost:.2f}", f"{deep_lost:.2f}")
        print("| " + " | ".join(row) + " |", flush=True)


if __name__ == "__main__":
    main()
