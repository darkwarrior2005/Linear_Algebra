# 🧮 Linear Algebra Visualizer

An interactive Python app for **seeing** linear algebra and vector calculus in 2D and 3D.
Type a matrix, a vector or a formula and watch what it does: grids bend, arrows turn,
eigenvectors hold their line, fluids flow, and particles move through fields that change with time.

There is no theory to read through. Every page is a playground with presets you can change.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py          # or:  python run.py
```

The app opens in your browser at <http://localhost:8501>.
Press **▶ Play** under a plot to animate it. Drag to rotate 3D views and scroll to zoom.

## One graph, everything on it

The app is a single workspace: **one large graph** in the middle, a **control panel** on the left,
and the **legend and all numbers underneath** the graph, so nothing covers the picture.
Every feature is a layer you switch on in the control panel. Layers combine, so you can, for example,
watch particles flow through a field while a matrix rotates the grid and your vectors.

| Layer (control panel) | What it adds to the graph |
|---|---|
| 🔀 **Matrix / transformation** | Applies a 2×2 / 3×3 matrix A to the whole space: grid, unit square/cube, unit circle/sphere, basis arrows, fan of unit vectors. Can play A as one smooth move or split into SVD (rotate → stretch → rotate), eigen (P⁻¹ → D → P), polar, QR or LU. Can compose with a second matrix B and finish with the inverse (undo). |
| ➡️ **Vectors** | Any number of vectors (an editable table). They ride along with the transformation. Optional: sum (tip-to-tail), linear combination with sliders, span, projection, cross product, Gram–Schmidt, power method, coordinates in another basis. |
| 🌟 **Eigenvectors & subspaces** | Eigenvector lines plus the column space, null space, row space and left null space of A. |
| 🎯 **Equations Ax = b** | Row picture (lines/planes), column picture (mixing columns to reach b), the solution, or the least-squares best fit. |
| 🧭 **Another basis B** | B's grid and basis vectors (one click to use A's eigenvectors). |
| 🌬️ **Vector field F(x, y, z, t)** | Your formula, or the matrix itself (F = A·x). Arrows coloured by strength, streamlines, and a background coloured by strength, divergence or curl. Time-dependent fields animate. |
| ⛰️ **Scalar function f** | Level sets / isosurfaces and gradient arrows. |
| 🌊 **Particles** | A blob, grid, random cloud or line of particles carried by the field over time, with trails; the blob shows how area changes; fixed points can be marked and classified. |
| 🔍 **Probe** | A point where you see flux in/out (divergence), a spinning paddle wheel (curl), the Jacobian's eigen-directions with a deformed little square, and ∇f. |
| 🛰️ **Moving point r(t)** | A curve with the moving point, velocity, acceleration, the T-N-B frame and the osculating circle. |
| ⏱️ **Time** | Time range, frames, and "let all of space flow with x′ = Ax" (the grid moves by e^{tA}). |

**▶ Play** animates either *the matrix* (step by step) or *time* (fields, particles, curve, probe).
*auto* chooses time when something time-based is switched on.

**Smooth transformations.** A matrix is grown out of the identity along its natural path, e^{tL} with
e^L = A. Rotations turn rigidly instead of shrinking, shears slide, and stretches grow. Matrices
without a real logarithm (reflections, projections, half-turns) are split into a rotation that turns
and a stretch that slides, where any flip or squash happens.

Under the graph, info cards show det, rank, trace, a plain-language description, the eigenvalue
table, the characteristic polynomial, vector lengths, dot products and angles, the solution of Ax = b
with every row-reduction step, subspace bases, symbolic divergence and curl, fixed points, the probe's
F, div, curl and Jacobian, and the curve's speed and curvature.
A **matrix calculator** for any size sits at the bottom. It can send a 2×2/3×3 result to the graph.

## Typing formulas

* Matrix and vector cells accept numbers **or expressions**: `sqrt(2)/2`, `cos(pi/6)`, `-1/3`, `2e-3`.
* Fields use `x, y, z` and time `t`, for example `-y + 0.1*x`, `sin(x)*exp(-t)` or `x^2 - y^2`.
* Functions: `sin cos tan asin acos atan atan2 sinh cosh tanh exp log ln log10 sqrt abs sign floor ceil min max where(cond, a, b) heaviside`. Constants: `pi`, `e`, `tau`.
* `^` means power, and `2x` means `2*x`.
* Input is checked against a whitelist before it runs, so only maths is accepted.

### Example: your own time-dependent field

Open **🌬️ Vector field**, switch it on, pick *(custom)* and type

```
Fx = -y + 0.5*sin(t)
Fy =  x - 0.2*y
```

Then switch on **🌊 Particles** (start as *blob*) and press ▶ Play. The blob of fluid is carried, stretched
and rotated by the field as it changes over time.

## Project layout

```
app.py                    Streamlit entry point
run.py                    `python run.py` launcher
linalg_viz/
  core/parsing.py         safe expression / matrix parser
  core/linalg.py          eigen, RREF, solve, subspaces, Gram–Schmidt, LU/QR/SVD, e^A, log A, smooth paths
  core/fields.py          vector/scalar fields, div/curl/grad/Laplacian, RK4 flows, fixed points, r(t)
  viz/plot.py             Plotly arrows, grids, shapes, quivers and animation helpers (2D & 3D)
  workspace/controls.py   the control panel → Settings
  workspace/scene.py      every layer of the single graph + animation frames
  workspace/panels.py     legend chips and info cards (outside the graph)
  workspace/calculator.py matrix calculator for any size
tests/                    unit tests for the maths + smoke tests switching on every layer and option
```

Run the tests with `pip install pytest && pytest`.
