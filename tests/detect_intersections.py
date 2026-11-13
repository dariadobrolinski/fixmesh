"""Detect self-intersecting face pairs in one or more meshes.

Run with: python tests/detect_intersections.py mesh.stl [another_mesh.obj ...]
"""

import argparse

import pymesh


def main():
    parser = argparse.ArgumentParser(
        description="Report self-intersecting face pairs in mesh files."
    )
    parser.add_argument(
        "mesh_paths",
        nargs="+",
        help="Path(s) to mesh files supported by PyMesh.",
    )
    parser.add_argument(
        "--show-pairs",
        action="store_true",
        help="Print the face indices for each intersecting pair.",
    )
    args = parser.parse_args()

    for mesh_path in args.mesh_paths:
        mesh = pymesh.load_mesh(mesh_path)
        intersections = pymesh.detect_self_intersection(mesh)
        print(f"{mesh_path}: {len(intersections)} self-intersecting face pair(s)")
        if args.show_pairs:
            for face_a, face_b in intersections:
                print(f"  faces {face_a} and {face_b}")


if __name__ == "__main__":
    main()