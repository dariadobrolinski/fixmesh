from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import open3d as o3d
import pymesh
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fixmesh.methods import push_apart_multi as PM  # noqa: E402


def _sphere(radius, center=(0, 0, 0), subdivisions=4):
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions,
                                        radius=radius)
    return pymesh.form_mesh(sphere.vertices + np.array(center, dtype=float),
                            sphere.faces)


def _crossings(results):
    surfaces = [(np.asarray(r.vertices), np.asarray(r.faces)) for r in results]
    vertices, faces, _, _ = PM._stack(surfaces)
    return len(PM._detect(vertices, faces))


def _as_surface(mesh):
    return np.asarray(mesh.vertices), np.asarray(mesh.faces, dtype=np.int64)


class NestingTests(unittest.TestCase):
    def test_small_sphere_is_inside_big_one(self):
        inside = PM._nesting([_as_surface(_sphere(1)), _as_surface(_sphere(2))])
        self.assertTrue(inside[0, 1])
        self.assertFalse(inside[1, 0])

    def test_side_by_side_spheres_are_not_nested(self):
        inside = PM._nesting([_as_surface(_sphere(1)),
                              _as_surface(_sphere(1, (1.5, 0, 0)))])
        self.assertFalse(inside.any())


class PushDirectionTests(unittest.TestCase):
    def setUp(self):
        # Surface 0 has faces 0-1, surface 1 has faces 2-3.
        self.faces = np.array([[0, 1, 2], [1, 2, 3], [4, 5, 6], [5, 6, 7]])
        self.labels = np.array([0, 0, 1, 1])

    def directions(self, pairs, inside):
        return PM._push_directions(np.array(pairs), self.faces, self.labels,
                                   np.array(inside))

    def test_nested_inner_goes_in_and_outer_goes_out(self):
        inward, outward = self.directions(
            [[0, 2]], [[False, True], [False, False]])
        self.assertEqual(np.nonzero(inward)[0].tolist(), [0, 1, 2])
        self.assertEqual(np.nonzero(outward)[0].tolist(), [4, 5, 6])

    def test_side_by_side_both_go_in(self):
        inward, outward = self.directions(
            [[0, 2]], [[False, False], [False, False]])
        self.assertEqual(np.nonzero(inward)[0].tolist(), [0, 1, 2, 4, 5, 6])
        self.assertFalse(outward.any())

    def test_self_crossing_goes_in_even_when_nested(self):
        inward, outward = self.directions(
            [[2, 3]], [[False, True], [False, False]])
        self.assertEqual(np.nonzero(inward)[0].tolist(), [4, 5, 6, 7])
        self.assertFalse(outward.any())


class RepairTests(unittest.TestCase):
    def test_nested_surfaces_separate_with_inner_inside(self):
        inner, outer = _sphere(1.0, (0.15, 0, 0)), _sphere(1.1)
        results = PM.push_apart_multi_repair([inner, outer], step=0.02,
                                             max_displacement=0.3)
        self.assertEqual(_crossings(results), 0)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(
            o3d.core.Tensor(results[1].vertices.astype(np.float32)),
            o3d.core.Tensor(results[1].faces.astype(np.uint32)))
        occupancy = scene.compute_occupancy(o3d.core.Tensor(
            results[0].vertices.astype(np.float32))).numpy()
        self.assertTrue(occupancy.all())

    def test_side_by_side_surfaces_separate(self):
        results = PM.push_apart_multi_repair(
            [_sphere(1.0), _sphere(1.0, (1.9, 0, 0))], step=0.02,
            max_displacement=0.3)
        self.assertEqual(_crossings(results), 0)

    def test_faces_are_kept_when_nothing_gets_flipped(self):
        meshes = [_sphere(1.0), _sphere(1.0, (1.9, 0, 0))]
        results = PM.push_apart_multi_repair(meshes, step=0.02,
                                             max_displacement=0.3,
                                             flip_squashed=False)
        for mesh, result in zip(meshes, results):
            self.assertTrue(np.array_equal(mesh.faces, result.faces))

    def test_surfaces_that_dont_cross_are_left_alone(self):
        meshes = [_sphere(1.0), _sphere(2.0)]
        results = PM.push_apart_multi_repair(meshes)
        for mesh, result in zip(meshes, results):
            self.assertTrue(np.allclose(mesh.vertices, result.vertices))

    def test_too_far_to_separate_stays_within_limit(self):
        meshes = [_sphere(1.0), _sphere(1.0, (1.0, 0, 0))]
        results = PM.push_apart_multi_repair(meshes, step=0.02,
                                             max_displacement=0.1)
        for mesh, result in zip(meshes, results):
            moved = np.linalg.norm(result.vertices - mesh.vertices, axis=1)
            self.assertLessEqual(moved.max(), 0.1 + 1e-9)

    def test_one_surface_is_refused(self):
        with self.assertRaises(ValueError):
            PM.push_apart_multi_repair([_sphere(1.0)])

    def test_command_saves_one_file_per_surface(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for name, mesh in (("a.ply", _sphere(1.0)),
                               ("b.ply", _sphere(1.0, (1.9, 0, 0)))):
                pymesh.save_mesh(str(folder / name), mesh)
            PM.main([str(folder / "a.ply"), str(folder / "b.ply"),
                     "-o", str(folder / "out"), "-s", "0.02", "-l", "0.3"])
            self.assertEqual(
                sorted(p.name for p in (folder / "out").iterdir()),
                ["push_apart_a.ply", "push_apart_b.ply"])


if __name__ == "__main__":
    unittest.main()
