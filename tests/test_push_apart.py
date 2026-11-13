from pathlib import Path
import sys
import unittest

import numpy as np
import pymesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fixmesh.methods import push_apart as PA  # noqa: E402


def _strip(num_triangles):
    """A flat strip of triangles in a row: triangle k is next to k-1 and k+1."""
    vertices = np.array([[x, y, 0.0] for x in range(num_triangles) for y in (0, 1)])
    faces = []
    for k in range(num_triangles - 1):
        a, b, c, d = 2 * k, 2 * k + 1, 2 * k + 2, 2 * k + 3
        faces += [[a, c, b], [b, c, d]]
    faces = np.array(faces)
    return faces, PA._vertex_adjacency(faces, len(vertices))


class NewCrossingTests(unittest.TestCase):
    def setUp(self):
        # The two halves of one long strip stand in for two walls.
        self.faces, self.adjacency = _strip(40)
        self.wall_b = len(self.faces) // 2
        self.ever = np.zeros(self.adjacency.shape[0], dtype=bool)
        # Face 0 on wall A once crossed face wall_b on wall B.
        self.ever[self.faces[[0, self.wall_b]].ravel()] = True

    def new(self, pairs):
        return PA._new_crossings(np.array(pairs), self.faces, self.ever,
                                 self.adjacency)

    def test_same_crossing_is_not_new(self):
        self.assertEqual(len(self.new([[0, self.wall_b]])), 0)

    def test_crossing_that_slid_one_triangle_is_not_new(self):
        self.assertEqual(len(self.new([[1, self.wall_b + 1]])), 0)

    def test_crossing_somewhere_fresh_is_new(self):
        self.assertEqual(len(self.new([[30, self.wall_b + 30]])), 1)

    def test_old_crosser_hitting_a_different_wall_is_new(self):
        # Face 0 has crossed before, but face 30 is far from any old crossing:
        # like one side of a thin gyrus pushed into the other side.
        self.assertEqual(len(self.new([[0, 30]])), 1)


class PushApartTests(unittest.TestCase):
    def test_small_meshes_end_without_crossings(self):
        for filename in ("two_spheres.ply", "three_spheres.ply"):
            with self.subTest(filename=filename):
                mesh = pymesh.load_mesh(str(ROOT / "tests/data" / filename))
                repaired = PA.push_apart_repair(mesh, units="edge")
                self.assertEqual(len(pymesh.detect_self_intersection(
                    pymesh.form_mesh(repaired.vertices, repaired.faces))), 0)

    def test_mm_units_match_edge_units_when_scaled(self):
        mesh = pymesh.load_mesh(str(ROOT / "tests/data/two_spheres.ply"))
        edge = PA._compute_median_edge_length(mesh)
        by_edge = PA.push_apart_repair(mesh, cut_leftovers=False, clearance=0,
                                       units="edge")
        by_mm = PA.push_apart_repair(mesh, cut_leftovers=False, clearance=0,
                                     units="mm", step=0.2 * edge,
                                     max_displacement=3.0 * edge)
        np.testing.assert_allclose(by_edge.vertices, by_mm.vertices, atol=1e-9)

    def test_rejects_unknown_units(self):
        mesh = pymesh.load_mesh(str(ROOT / "tests/data/two_spheres.ply"))
        with self.assertRaises(ValueError):
            PA.push_apart_repair(mesh, units="inches")


class OutputPathTests(unittest.TestCase):
    def test_default_is_next_to_input(self):
        self.assertEqual(PA._output_path("data/brain.stl", None),
                         "data/push_apart_brain.stl")

    def test_folder_keeps_default_name(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(PA._output_path("data/brain.stl", folder),
                             f"{folder}/push_apart_brain.stl")

    def test_file_name_is_used_as_is(self):
        self.assertEqual(PA._output_path("data/brain.stl", "fixed.stl"),
                         "fixed.stl")


if __name__ == "__main__":
    unittest.main()
