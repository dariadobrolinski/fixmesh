import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pymesh


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fixmesh_cut", ROOT / "fixmesh/cut.py")
CUT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CUT)


class CutRepairTests(unittest.TestCase):
    def test_smoothing_blends_movement_into_neighboring_vertices(self):
        mesh = pymesh.load_mesh(str(ROOT / "tests/data/two_spheres.ply"))
        resolved = CUT._resolve_and_separate_components(mesh)
        intersections = pymesh.detect_self_intersection(resolved)
        target = CUT._compute_median_edge_length(mesh) * 1.6

        unsmoothed = CUT._separate_intersecting_components(
            resolved,
            intersections,
            target,
            smooth=False,
        )
        smoothed = CUT._separate_intersecting_components(
            resolved,
            intersections,
            target,
            smooth=True,
            smoothing_rings=3,
        )

        unsmoothed_movement = np.linalg.norm(
            unsmoothed.vertices - resolved.vertices, axis=1
        )
        smoothed_movement = np.linalg.norm(
            smoothed.vertices - resolved.vertices, axis=1
        )
        self.assertGreater(
            np.count_nonzero(smoothed_movement > 1e-12),
            np.count_nonzero(unsmoothed_movement > 1e-12),
        )
        self.assertLess(smoothed_movement.max(), unsmoothed_movement.max())

    def test_multi_component_repairs_are_closed_and_intersection_free(self):
        cases = (
            ("sphere_cube.ply", 2),
            ("two_spheres.ply", 2),
            ("three_spheres.ply", 3),
        )
        for filename, expected_components in cases:
            with self.subTest(filename=filename):
                mesh = pymesh.load_mesh(str(ROOT / "tests/data" / filename))
                original_face_count = mesh.num_faces

                repaired = CUT.cut_repair(
                    mesh,
                    strategy="geometry_preserving",
                    max_passes=20,
                )
                repaired_pymesh = pymesh.form_mesh(
                    repaired.vertices, repaired.faces
                )

                self.assertEqual(
                    len(pymesh.detect_self_intersection(repaired_pymesh)), 0
                )
                self.assertEqual(
                    repaired_pymesh.num_surface_components,
                    expected_components,
                )
                self.assertTrue(repaired.is_watertight)
                self.assertGreaterEqual(
                    repaired_pymesh.num_faces, original_face_count
                )

    def test_rejects_unknown_strategy(self):
        mesh = pymesh.load_mesh(str(ROOT / "tests/data/sphere_cube.ply"))
        with self.assertRaisesRegex(ValueError, "unknown repair strategy"):
            CUT.cut_repair(mesh, strategy="not-a-strategy")


if __name__ == "__main__":
    unittest.main()
