"""
Check the outputs of the cut/repair pipeline (fixmesh/cut.py).

This script checks the number of components (i.e both brain hemispheres) and
checks if there are any self interescting face pairs using
pymesh.detect_self_intersections().

To run:
tests/validate_mesh.py <mesh_path>
"""

import argparse
import pymesh


def main():
    parser = argparse.ArgumentParser(
        description="Report connected-component count and self-intersecting "
        "face-pair count for a mesh."
    )
    parser.add_argument("mesh_path", help="Path to the mesh to validate.")
    args = parser.parse_args()

    mesh = pymesh.load_mesh(args.mesh_path)

    num_intersections = len(pymesh.detect_self_intersection(mesh))

    print(f"components              : {mesh.num_surface_components}")
    print(f"self-intersecting pairs : {num_intersections}")


if __name__ == "__main__":
    main()
