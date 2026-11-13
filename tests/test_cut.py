import importlib.util
from pathlib import Path
import unittest

import pymesh


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fixmesh_cut", ROOT / "fixmesh/methods/cut.py")
CUT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CUT)


class CutRepairTests(unittest.TestCase):
    def test_multi_component_repairs_are_closed_and_intersection_free(self):
        cases = (
            ("sphere_cube.ply", 2),
            ("two_spheres.ply", 2),
            ("three_spheres.ply", 3),
        )
        for filename, expected_components in cases:
            with self.subTest(filename=filename):
                mesh = pymesh.load_mesh(str(ROOT / "tests/data" / filename))
                repaired = CUT.cut_repair(mesh, max_passes=20)
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

    def test_rejects_unknown_fairing(self):
        mesh = pymesh.load_mesh(str(ROOT / "tests/data/sphere_cube.ply"))
        with self.assertRaisesRegex(ValueError, "unknown fairing mode"):
            CUT.cut_repair(mesh, fairing="not-a-mode")


if __name__ == "__main__":
    unittest.main()
