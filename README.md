# fixmesh

This is a tool where you can try different 3D mesh repair methods. It wraps multiple mesh processing libraries such as PyMesh, PyMeshFix, MeshLib, and more.


## Installation

See `PyMesh_Installation_Guide.md` for PyMesh installation. After creating a conda environment, install all requried libraries specified in `pymesh_env.yml`.

```bash
git clone https://github.com/yourusername/fixmesh.git
cd fixmesh
```

## Repair methods

```python
result = fixmesh.cut_repair(mesh)          # cut the crossing faces and patch the holes
result = fixmesh.push_apart_repair(mesh)   # push crossing walls apart instead of cutting
```

See `docs/methods_comparison.md` for how the methods compare.

## Documentation
See https://kenichi-maeda.github.io/fixmesh/.

## Visualization
See https://kenichi-maeda.github.io/meshViewer/. (Merging Neighboring Meshes)<br>
See https://kenichi-maeda.github.io/meshViewer2/ (Detaching Enclosed Meshes)<br>
See https://kenichi-maeda.github.io/meshViewer3/ (Detaching Neighboring Meshes).

## Acknowledgement
I received assistance from ChatGPT for coding.
