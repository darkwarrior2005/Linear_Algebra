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

## What's inside

| Section | Page | What you can do |
|---|---|---|
| **Vectors & matrices** | Vectors | Add, scale and combine arrows tip-to-tail. Show span (line/plane/space) and linear independence. Show dot product, projection, angle, cross product and parallelogram area. |
| | Linear transformations | Animate any 2×2 / 3×3 matrix acting on the grid, the unit square/cube, the unit circle/sphere and your own vector. Shows det, rank and a plain-language description (rotation, shear, projection…). |
| | Composition & inverse | Watch *B then A* (= AB) step by step and see that AB ≠ BA. A⁻¹ undoes A. Compare A with Aᵀ. |
| | Systems Ax = b | Row picture (lines/planes meeting), column picture (mixing columns to reach b), least squares when there is no solution, and row reduction step by step. |
| | Subspaces & rank | Column space, null space, row space, left null space, rank–nullity, and a cloud of points being squashed. |
| | Change of basis | Coordinates in a new basis with its own grid, the same map in another basis (B⁻¹AB), and a one-click eigenbasis. |
| | Gram–Schmidt & projections | Orthogonalisation step by step (→ QR), and projection onto a line/plane with the error vector. |
| **Eigen & decompositions** | Eigenvectors & eigenvalues | A fan of test vectors: only the eigenvectors stay on their lines. Hunt for eigen-directions by hand, run the power method, plot the characteristic polynomial and see complex eigenvalues. |
| | Decompositions | SVD (rotate → stretch → rotate), eigendecomposition PDP⁻¹, polar QS, QR and LU, each animated as a sequence of simple moves. |
| | Matrix calculator | Any size: det, inverse, RREF with every row operation, eigen, SVD, QR, LU, null/column space, Aⁿ, e^A, pseudo-inverse, norms, solve Ax = b, and a picture for 2×2/3×3 matrices. |
| **Fields & time evolution** | Vector fields | Type **F(x, y, z, t)**. Draw arrows coloured by strength and streamlines over a magnitude/divergence/curl background. Time-dependent fields animate. Divergence, curl and the Jacobian are worked out symbolically. |
| | Differential operators | **Gradient** (contours + 3D surface + directional derivative), **divergence** (flux through a probe circle/sphere), **curl** (a spinning paddle wheel), **Laplacian** (neighbour average), the **Jacobian** as the local linear map, and checks of curl(∇f) = 0 and div(curl F) = 0. |
| | Flows & particles | Release particles (grid, random, a blob of fluid, a line, your own points) into any field, including time-dependent ones, and watch them move with trails. Tracks the blob's area and finds and classifies fixed points. |
| | Linear dynamics x′ = Ax | Phase portraits, eigen-lines, a trace–determinant map, and space flowing under e^{tA}. Eigenvalues are read off as growth, decay and oscillation. |
| | Motion along a curve r(t) | Animated position, velocity, acceleration, Frenet frame T-N-B, osculating circle, speed, curvature and torsion. |

## Typing formulas

* Matrix and vector cells accept numbers **or expressions**: `sqrt(2)/2`, `cos(pi/6)`, `-1/3`, `2e-3`.
* Fields use `x, y, z` and time `t`, for example `-y + 0.1*x`, `sin(x)*exp(-t)` or `x^2 - y^2`.
* Functions: `sin cos tan asin acos atan atan2 sinh cosh tanh exp log ln log10 sqrt abs sign floor ceil min max where(cond, a, b) heaviside`. Constants: `pi`, `e`, `tau`.
* `^` means power, and `2x` means `2*x`.
* Input is checked against a whitelist before it runs, so only maths is accepted.

### Example: your own time-dependent field

On **Flows & particles**, pick *(custom)* and type

```
Fx = -y + 0.5*sin(t)
Fy =  x - 0.2*y
```

Then choose *blob* as the starting positions and press ▶ Play. The blob of fluid is carried, stretched
and rotated by the field as it changes over time.

## Project layout

```
app.py                  Streamlit entry point (sidebar navigation)
run.py                  `python run.py` launcher
linalg_viz/
  core/parsing.py       safe expression / matrix parser
  core/linalg.py        eigen, RREF, solve, subspaces, Gram–Schmidt, LU/QR/SVD, e^A …
  core/fields.py        vector/scalar fields, div/curl/grad/Laplacian, RK4 flows, fixed points, r(t)
  viz/plot.py           Plotly arrows, grids, shapes, quivers and animation helpers (2D & 3D)
  viz/scenes.py         reusable "space being transformed" animations
  pages/*.py            one module per page
tests/                  unit tests for the maths + smoke tests rendering every page
```

Run the tests with `pip install pytest && pytest`.
