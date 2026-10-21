"""Ready-made matrices, vector fields and curves (entries may be expressions)."""

MATRICES_2D = {
    "Identity": [[1, 0], [0, 1]],
    "Rotation 45°": [["cos(pi/4)", "-sin(pi/4)"], ["sin(pi/4)", "cos(pi/4)"]],
    "Rotation 90°": [[0, -1], [1, 0]],
    "Scale (2, 0.5)": [[2, 0], [0, 0.5]],
    "Uniform scale ×1.5": [[1.5, 0], [0, 1.5]],
    "Shear (horizontal)": [[1, 1], [0, 1]],
    "Shear (vertical)": [[1, 0], [1, 1]],
    "Reflect across x-axis": [[1, 0], [0, -1]],
    "Reflect across y = x": [[0, 1], [1, 0]],
    "Project onto x-axis": [[1, 0], [0, 0]],
    "Squash onto a line (rank 1)": [[1, 2], [0.5, 1]],
    "Symmetric (eigen-friendly)": [[2, 1], [1, 2]],
    "Stretch along eigenvectors": [[3, 1], [0, 2]],
    "Rotation + scaling (complex λ)": [[1, -1], [1, 1]],
    "Defective (one eigen-direction)": [[1, 1], [0, 1]],
    "Fibonacci": [[1, 1], [1, 0]],
}

MATRICES_3D = {
    "Identity": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "Rotate 45° about z": [["cos(pi/4)", "-sin(pi/4)", 0], ["sin(pi/4)", "cos(pi/4)", 0], [0, 0, 1]],
    "Rotate 90° about x": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
    "Scale (2, 1, 0.5)": [[2, 0, 0], [0, 1, 0], [0, 0, 0.5]],
    "Shear x by z": [[1, 0, 1], [0, 1, 0], [0, 0, 1]],
    "Reflect through xy-plane": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
    "Project onto xy-plane": [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
    "Squash onto a line (rank 1)": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    "Symmetric": [[2, 1, 0], [1, 2, 1], [0, 1, 2]],
    "Upper triangular": [[2, 1, 1], [0, 1.5, 1], [0, 0, 0.5]],
    "Spiral (rotation + stretch)": [["cos(pi/6)", "-sin(pi/6)", 0], ["sin(pi/6)", "cos(pi/6)", 0], [0, 0, 1.5]],
    "Permutation (cycle axes)": [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
}

MATRICES = {2: MATRICES_2D, 3: MATRICES_3D}

LINEAR_SYSTEMS_2D = {
    "Saddle": [[1, 0], [0, -1]],
    "Stable node": [[-2, 0], [0, -1]],
    "Unstable node": [[2, 1], [0, 1]],
    "Stable spiral": [[-0.3, -1], [1, -0.3]],
    "Unstable spiral": [[0.2, -1], [1, 0.2]],
    "Center (harmonic oscillator)": [[0, 1], [-1, 0]],
    "Damped oscillator": [[0, 1], [-1, -0.4]],
    "Degenerate node": [[-1, 1], [0, -1]],
    "Line of fixed points": [[-1, 0], [0, 0]],
    "Saddle (tilted)": [[1, 2], [2, 1]],
}

LINEAR_SYSTEMS_3D = {
    "Stable spiral + decay": [[-0.2, -1, 0], [1, -0.2, 0], [0, 0, -0.5]],
    "Saddle-spiral": [[-0.1, -1, 0], [1, -0.1, 0], [0, 0, 0.4]],
    "Stable sink": [[-1, 0, 0], [0, -0.5, 0], [0, 0, -0.3]],
    "Rotation about z": [[0, -1, 0], [1, 0, 0], [0, 0, 0]],
    "Coupled": [[-0.5, 1, 0], [-1, -0.5, 1], [0, -1, -0.5]],
}

LINEAR_SYSTEMS = {2: LINEAR_SYSTEMS_2D, 3: LINEAR_SYSTEMS_3D}

# Vector fields: (components, recommended view size, description)
FIELDS_2D = {
    "Rotation (vortex)": (["-y", "x"], 3, "Pure rotation: divergence 0, curl 2 everywhere."),
    "Source": (["x", "y"], 3, "Everything flows outward: divergence 2, curl 0."),
    "Sink": (["-x", "-y"], 3, "Everything flows inward: divergence −2."),
    "Saddle": (["x", "-y"], 3, "In along y, out along x."),
    "Shear flow": (["y", "0"], 3, "Layers slide past each other: zero divergence but nonzero curl."),
    "Spiral sink": (["-x - y", "x - y"], 3, "Rotation plus contraction."),
    "Point vortex": (["-y/(x^2+y^2+0.05)", "x/(x^2+y^2+0.05)"], 3, "Swirl strongest near the centre, curl ≈ 0 away from it."),
    "Dipole": (["(x^2 - y^2)/((x^2+y^2)^2+0.1)", "2*x*y/((x^2+y^2)^2+0.1)"], 2.5, "Field lines loop from one side to the other."),
    "Pendulum (phase space)": (["y", "-sin(x)"], 6, "x = angle, y = angular velocity. Closed loops = swinging, waves = spinning."),
    "Damped pendulum": (["y", "-sin(x) - 0.3*y"], 6, "Friction makes every motion spiral into a resting point."),
    "Van der Pol oscillator": (["y", "(1 - x^2)*y - x"], 4, "Every trajectory converges to one closed loop (a limit cycle)."),
    "Lotka–Volterra (predator/prey)": (["x*(1.5 - y)", "y*(x - 1)"], 4, "Populations cycle forever around an equilibrium."),
    "Gradient of a hill": (["-2*x*exp(-(x^2+y^2))", "-2*y*exp(-(x^2+y^2))"], 2.5, "−∇ of a Gaussian bump: flow goes downhill."),
    "Rotating field (time-dependent)": (["cos(t) - 0.3*x", "sin(t) - 0.3*y"], 3, "A uniform wind whose direction turns with time."),
    "Pulsating source (time-dependent)": (["x*sin(t)", "y*sin(t)"], 3, "Alternately breathes out and in."),
    "Double gyre (time-dependent)": (
        ["-pi*0.1*sin(pi*(0.25*sin(0.6*t)*x^2 + (1-0.5*sin(0.6*t))*x))*cos(pi*y)",
         "pi*0.1*cos(pi*(0.25*sin(0.6*t)*x^2 + (1-0.5*sin(0.6*t))*x))*sin(pi*y)*(0.5*sin(0.6*t)*x + 1 - 0.5*sin(0.6*t))"],
        2.2, "Classic ocean-mixing model: two swirling cells whose boundary sloshes in time (best on x∈[0,2], y∈[0,1])."),
    "Travelling wave (time-dependent)": (["1", "sin(x - 2*t)"], 4, "A wave pattern moving to the right."),
}

FIELDS_3D = {
    "Rotation about z": (["-y", "x", "0"], 3, "Spins around the z-axis: curl = (0, 0, 2)."),
    "Source": (["x", "y", "z"], 3, "Radially outward: divergence 3."),
    "Helical flow": (["-y", "x", "0.5"], 3, "Rotation plus a steady rise: particles move on helices."),
    "Saddle": (["x", "y", "-2*z"], 3, "Out in the xy-plane, in along z; divergence 0."),
    "Lorenz system": (["10*(y - x)", "x*(28 - z) - y", "x*y - 8/3*z"], 30, "The famous chaotic butterfly attractor."),
    "ABC flow": (["sin(z) + cos(y)", "sin(x) + cos(z)", "sin(y) + cos(x)"], 3.2, "A divergence-free flow with chaotic streamlines."),
    "Vortex ring-ish": (["-y + 0.2*x*z", "x + 0.2*y*z", "0.5 - 0.1*(x^2+y^2)"], 3, "Swirl with up-flow in the middle and down-flow outside."),
    "Rössler system": (["-y - z", "x + 0.2*y", "0.2 + z*(x - 5.7)"], 15, "A chaotic attractor shaped like a folded band."),
    "Rotating axis (time-dependent)": (["-y*cos(t)", "x*cos(t) - z*sin(t)", "y*sin(t)"], 3, "The spin axis itself tilts over time."),
}

FIELDS = {2: FIELDS_2D, 3: FIELDS_3D}

# Fields that look best on a particular window (xmin, xmax, ymin, ymax).
FIELD_DOMAINS = {"Double gyre (time-dependent)": (0.0, 2.0, 0.0, 1.0)}

SCALARS_2D = {
    "Bowl  x² + y²": "x^2 + y^2",
    "Saddle  x² − y²": "x^2 - y^2",
    "Gaussian hill": "exp(-(x^2 + y^2))",
    "Two hills": "exp(-((x-1)^2 + y^2)) + 0.8*exp(-((x+1)^2 + (y-0.5)^2))",
    "Waves  sin(x)·cos(y)": "sin(x)*cos(y)",
    "Ripple": "sin(2*sqrt(x^2+y^2))",
    "Tilted plane": "0.5*x + 0.25*y",
    "Moving bump (time-dependent)": "exp(-((x - cos(t))^2 + (y - sin(t))^2))",
}

SCALARS_3D = {
    "Distance²  x²+y²+z²": "x^2 + y^2 + z^2",
    "Gaussian blob": "exp(-(x^2 + y^2 + z^2))",
    "Saddle  x² + y² − z²": "x^2 + y^2 - z^2",
    "Waves": "sin(x)*cos(y)*sin(z)",
}

SCALARS = {2: SCALARS_2D, 3: SCALARS_3D}

CURVES_2D = {
    "Circle": (["cos(t)", "sin(t)"], (0, "2*pi")),
    "Ellipse": (["2*cos(t)", "sin(t)"], (0, "2*pi")),
    "Spiral": (["0.3*t*cos(t)", "0.3*t*sin(t)"], (0, "4*pi")),
    "Lissajous 3:2": (["sin(3*t)", "sin(2*t)"], (0, "2*pi")),
    "Cycloid (rolling wheel)": (["t - sin(t)", "1 - cos(t)"], (0, "4*pi")),
    "Projectile": (["3*t", "4*t - 4.9*t^2/2"], (0, "1.63")),
    "Cardioid": (["(1 - cos(t))*cos(t)", "(1 - cos(t))*sin(t)"], (0, "2*pi")),
    "Figure eight": (["sin(t)", "sin(t)*cos(t)"], (0, "2*pi")),
}

CURVES_3D = {
    "Helix": (["cos(t)", "sin(t)", "0.2*t"], (0, "4*pi")),
    "Conical spiral": (["0.2*t*cos(t)", "0.2*t*sin(t)", "0.2*t"], (0, "6*pi")),
    "Trefoil knot": (["sin(t) + 2*sin(2*t)", "cos(t) - 2*cos(2*t)", "-sin(3*t)"], (0, "2*pi")),
    "Viviani's curve": (["1 + cos(t)", "sin(t)", "2*sin(t/2)"], (0, "4*pi")),
    "Toroidal spiral": (["(2 + 0.5*cos(8*t))*cos(t)", "(2 + 0.5*cos(8*t))*sin(t)", "0.5*sin(8*t)"], (0, "2*pi")),
    "Projectile 3D": (["2*t", "1.5*t", "5*t - 4.9*t^2/2"], (0, "2.04")),
}

CURVES = {2: CURVES_2D, 3: CURVES_3D}
