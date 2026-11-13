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

## Command line

Install the `push_apart` command into the conda environment once (with
`pymesh_env` active; `--no-deps` keeps pip from replacing the source-built
PyMesh):

```bash
pip install -e . --no-deps
```

Then:

```bash
push_apart brain.stl                    # saves push_apart_brain.stl next to it
push_apart brain.stl results/           # saves results/push_apart_brain.stl
push_apart brain.stl results/fixed.stl  # saves exactly there
push_apart brain.stl -l 2 -s 0.1        # 2 mm limit, 0.1 mm step
push_apart brain.stl -g                 # also leave a 0.25 mm gap between touching walls
push_apart brain.stl -u                 # undo pushes that cross somewhere new
push_apart -h                           # all options
```

## Documentation
See https://kenichi-maeda.github.io/fixmesh/.

## Visualization
See https://kenichi-maeda.github.io/meshViewer/. (Merging Neighboring Meshes)<br>
See https://kenichi-maeda.github.io/meshViewer2/ (Detaching Enclosed Meshes)<br>
See https://kenichi-maeda.github.io/meshViewer3/ (Detaching Neighboring Meshes).

## Acknowledgement
I received assistance from ChatGPT for coding.
