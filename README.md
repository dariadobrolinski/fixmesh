# fixmesh

This is a tool where you can try different 3D mesh repair methods. It wraps multiple mesh processing libraries such as PyMesh, PyMeshFix, MeshLib, and more.


## Installation

See `PyMesh_Installation_Guide.md` for PyMesh installation. After creating a conda environment, install all requried libraries specified in `pymesh_env.yml`.

```bash
git clone https://github.com/yourusername/fixmesh.git
cd fixmesh
```

## Geometry-preserving cut repair

`cut_repair` now preserves source faces by splitting triangles at intersection
curves and moving only the local colliding regions. It does not delete small
fragments or place flat caps over the cuts.

```python
result = fixmesh.cut_repair(mesh, strategy="geometry_preserving")
```

The previous cut-and-cap behavior remains available for comparisons:

```python
result = fixmesh.cut_repair(mesh, strategy="legacy")
```

The geometry-preserving strategy raises an error instead of returning a mesh
that still has intersections, has open boundaries, or changes the number of
components.

## Documentation
See https://kenichi-maeda.github.io/fixmesh/.

## Visualization
See https://kenichi-maeda.github.io/meshViewer/. (Merging Neighboring Meshes)<br>
See https://kenichi-maeda.github.io/meshViewer2/ (Detaching Enclosed Meshes)<br>
See https://kenichi-maeda.github.io/meshViewer3/ (Detaching Neighboring Meshes).

## Acknowledgement
I received assistance from ChatGPT for coding.
