"""
Featured image of the manuscript "Appellian Neural Networks".

The image is drawn for the journal home page: 1200 x 675 pixels, full frame,
no text. Every object comes from the saved results in SIMULATIONS.

  - The pendulum is one closed-loop forward simulation with ANNet (trajectory
    37). The bright pose is the state at 1.04 s. The fainter poses are the
    states of the preceding 1.04 s at intervals of 52 ms, the shaded areas are
    the areas swept by the two links in that time, the tapered trails are the
    paths of the two masses in that time, and the thin lines are their paths
    over the full 5 s.
  - The mesh is the objective that the forward solver minimizes at the state
    of the bright pose, evaluated with the trained network over the two joint
    accelerations within the solver bound (learned Appell energy plus the
    gravitational term plus the regularization term). Blue lines run along the
    acceleration of joint 1 and red lines along the acceleration of joint 2.
    The disc is the minimum of this objective and the rings around it are
    level sets, drawn at their own height.
  - The view looks along the valley of the objective, which is the direction
    of least curvature at the minimum, taken from the network by automatic
    differentiation.

The mesh fades with the distance from the minimum, and a veil dims it behind
the pendulum. Both are single gradient shapes (in the groups "light_falloff"
and "swept_areas") and remove no data. The script writes a dark and a light
version. All shapes are plain paths, circles and gradients in named groups,
with no filters, masks or clipping paths.

Run:  python draw_featured_image.py
"""

import os

import numpy as np
import torch
import torch.nn as nn
from contourpy import contour_generator

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "..", "..", "..", "SIMULATIONS")

# Simulation and solver settings of the manuscript
N_SAMPLES = 16000
SIM_FILE = "simulation_results_50trials_N16_B15_A10.pt"
M1 = M2 = L1 = L2 = 1.0
GRAVITY = 9.81
BOUND = 15.0
LAMBDA_REG = 0.1

# Trajectory and time window
TRIAL = 37
SWEEP_FIRST = 0       # first time step of the sweep
STATE_STEP = 520      # time step of the bright pose and of the objective (1.04 s)
POSE_EVERY = 26       # time steps between the fainter poses (52 ms)

# Frame and camera
WIDTH, HEIGHT = 1200, 675
PIVOT = (640.0, 42.0)         # pendulum pivot (pixels)
METRE = 250.0                 # pixels per metre
MESH_ORIGIN = (640.0, 448.0)  # image position of zero acceleration at the lowest objective value
MESH_ZOOM = 460.0             # pixels per 15 rad/s^2 at the depth of the origin
MESH_RISE = 1.0               # height of the objective range, in units of 15 rad/s^2
ELEVATION = 20.0              # degrees above the acceleration plane
CAMERA_DISTANCE = 3.2         # in units of 15 rad/s^2
MESH_EVERY = 4                # grid points between mesh lines (0.5 rad/s^2)
N_GRID = 241
RING_LEVELS = (np.arange(1, 6) / 5.0) ** 2 * 0.045   # level sets, as fractions of the objective range
RING_OPACITIES = np.linspace(0.50, 0.08, 5)
FALLOFF_RADIUS = 1250.0       # horizontal reach of the fade of the mesh (pixels)
FALLOFF_FLATTEN = 0.61        # height of the fade ellipse relative to its width
GROUND_RADIUS = 760.0         # radius of the darker centre of the background (pixels)

THEMES = {
    "dark": dict(
        ground_edge="#29303C", ground_centre="#161C27", falloff=(0.0, 0.30, 0.90, 1.0),
        blue="#3FA4EA", red="#EE5468", blue_core="#BFE6FF", red_core="#FFD0D5",
        mesh_opacity=0.78, glow=1.0, veil=0.55, minimum="#FFFFFF", minimum_halo="#FFE9C9", pivot="#FFFFFF",
    ),
    "light": dict(
        ground_edge="#FFFFFF", ground_centre="#FFFFFF", falloff=(0.0, 0.25, 0.85, 1.0),
        blue="#0072B2", red="#B2182B", blue_core="#0072B2", red_core="#B2182B",
        mesh_opacity=0.60, glow=0.55, veil=0.55, minimum="#1A1A1A", minimum_halo="#1A1A1A", pivot="#1A1A1A",
    ),
}


class GibbsAppellNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 225), nn.GELU(),
            nn.Linear(225, 61), nn.GELU(),
            nn.Linear(61, 91), nn.GELU(),
            nn.Linear(91, 99), nn.GELU(),
            nn.Linear(99, 1),
        )

    def forward(self, x):
        return self.net(x)


def load():
    sim = torch.load(os.path.join(SIM, SIM_FILE), map_location="cpu")
    stats = torch.load(os.path.join(SIM, f"stats_{N_SAMPLES}.pt"), map_location="cpu")
    model = GibbsAppellNet()
    model.load_state_dict(torch.load(os.path.join(SIM, f"model_{N_SAMPLES}.pth"), map_location="cpu"))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return sim, stats, model


def gravity_vector(q):
    g1 = GRAVITY * (L1 * (M1 + M2) * torch.sin(q[0]) + L2 * M2 * torch.sin(q[0] + q[1]))
    g2 = GRAVITY * L2 * M2 * torch.sin(q[0] + q[1])
    return torch.stack((g1, g2))


def make_objective(model, stats, q, dq, prev_ddq):
    """Objective of the forward solver as a function of the accelerations."""
    g_vec = gravity_vector(q)

    def objective(ddq):
        ddq = ddq.reshape(-1, 2)
        x = torch.cat((q.expand(len(ddq), 2), dq.expand(len(ddq), 2), ddq), dim=1)
        s = model((x - stats["X_mean"]) / (stats["X_std"] + 1e-8)).squeeze(-1)
        s = s * stats["S_std"] + stats["S_mean"]
        return s + ddq @ g_vec + LAMBDA_REG * (ddq - prev_ddq).square().sum(dim=1)

    return objective


def minimize(objective, start):
    ddq = start.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([ddq], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = objective(ddq).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return ddq.detach()


# ----------------------------------------------------------------- SVG helpers
def points_d(points):
    return " L".join(f"{x:.1f},{y:.1f}" for x, y in points)


def line(points, colour, width, opacity=1.0):
    return (f'<path d="M{points_d(points)}" fill="none" stroke="{colour}" stroke-width="{width:.2f}" '
            f'stroke-linecap="round" stroke-linejoin="round" opacity="{opacity:.3f}"/>')


def shape(points, fill, opacity=1.0, name=None):
    label = f' id="{name}"' if name else ""
    alpha = "" if fill.startswith("url") else f' opacity="{opacity:.3f}"'
    return f'<path{label} d="M{points_d(points)} Z" fill="{fill}"{alpha}/>'


def disc(x, y, r, fill, name=None):
    label = f' id="{name}"' if name else ""
    return f'<circle{label} cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}"/>'


def group(name, items, opacity=None):
    alpha = "" if opacity is None else f' opacity="{opacity:.3f}"'
    return f'<g id="{name}"{alpha}>\n' + "\n".join(items) + "\n</g>"


def stops(entries):
    return "".join(f'<stop offset="{o}" stop-color="{c}" stop-opacity="{a}"/>' for o, c, a in entries)


def radial_gradient(name, entries, cx=0.5, cy=0.5, r=0.5):
    return f'<radialGradient id="{name}" cx="{cx}" cy="{cy}" r="{r}">{stops(entries)}</radialGradient>'


def linear_gradient(name, start, end, entries):
    return (f'<linearGradient id="{name}" gradientUnits="userSpaceOnUse" x1="{start[0]:.1f}" y1="{start[1]:.1f}" '
            f'x2="{end[0]:.1f}" y2="{end[1]:.1f}">{stops(entries)}</linearGradient>')


def glowing_line(points, core, halo, width, strength):
    """A line with a soft halo, built from three strokes."""
    return [line(points, halo, 5.0 * width, 0.07 * strength),
            line(points, halo, 2.6 * width, 0.16 * strength),
            line(points, core, width)]


def tapered_trail(points, head_width, power=1.0):
    """Closed outline of a trail that widens from zero at its start to head_width at its end."""
    p = np.asarray(points, dtype=float)
    tangent = np.gradient(p, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.stack((-tangent[:, 1], tangent[:, 0]), axis=1)
    half = 0.5 * head_width * np.linspace(0.0, 1.0, len(p))[:, None] ** power
    return np.vstack((p + normal * half, (p - normal * half)[::-1]))


# ------------------------------------------------------------------- objective
def objective_layers(sim, stats, model, theme, defs):
    q = sim["all_pred_q"][TRIAL, STATE_STEP]
    dq = sim["all_pred_dq"][TRIAL, STATE_STEP]
    prev_ddq = sim["all_pred_ddq"][TRIAL, STATE_STEP - 1]
    objective = make_objective(model, stats, q, dq, prev_ddq)

    axis = torch.linspace(-BOUND, BOUND, N_GRID)
    a1, a2 = torch.meshgrid(axis, axis, indexing="ij")
    with torch.no_grad():
        J = objective(torch.stack((a1.reshape(-1), a2.reshape(-1)), dim=1)).reshape(N_GRID, N_GRID)
    i_min = np.unravel_index(int(J.argmin()), J.shape)
    minimum = minimize(objective, torch.stack((axis[i_min[0]], axis[i_min[1]])))
    j_min, j_max = float(objective(minimum)), float(J.max())

    # The valley runs along the direction of least curvature. The camera looks
    # along it from the end that is nearer to the minimum.
    hessian = torch.autograd.functional.hessian(lambda a: objective(a).sum(), minimum).numpy()
    curvature, directions = np.linalg.eigh(hessian)
    valley = directions[:, 0] * np.sign(directions[:, 0] @ minimum.numpy())
    azimuth = np.arctan2(valley[0], valley[1])
    elevation = np.deg2rad(ELEVATION)

    def project(u, v, z):
        """Accelerations (rad/s^2) and normalized objective to image coordinates and scale."""
        x = np.asarray(u, dtype=float) / BOUND
        y = np.asarray(v, dtype=float) / BOUND
        z = np.asarray(z, dtype=float) * MESH_RISE
        right = x * np.cos(azimuth) - y * np.sin(azimuth)
        away = -(x * np.sin(azimuth) + y * np.cos(azimuth))
        up = z * np.cos(elevation) + away * np.sin(elevation)
        depth = away * np.cos(elevation) - z * np.sin(elevation)
        scale = CAMERA_DISTANCE / (CAMERA_DISTANCE + depth)
        image = np.stack((MESH_ORIGIN[0] + MESH_ZOOM * scale * right,
                          MESH_ORIGIN[1] - MESH_ZOOM * scale * up), axis=-1)
        return image, scale

    def normalized(j):
        return (np.asarray(j, dtype=float) - j_min) / (j_max - j_min)

    A1, A2, Jn = a1.numpy(), a2.numpy(), J.numpy()
    every_other = slice(None, None, 2)
    lines_1, lines_2 = [], []
    for i in range(0, N_GRID, MESH_EVERY):
        border = i in (0, N_GRID - 1)
        opacity = 0.95 if border else theme["mesh_opacity"]
        image, scale = project(A1[every_other, i], A2[every_other, i], normalized(Jn[every_other, i]))
        lines_1.append(line(image, theme["blue"], (1.6 if border else 0.9) * scale.mean() ** 1.5, opacity))
        image, scale = project(A1[i, every_other], A2[i, every_other], normalized(Jn[i, every_other]))
        lines_2.append(line(image, theme["red"], (1.6 if border else 0.9) * scale.mean() ** 1.5, opacity))

    rings = []
    contours = contour_generator(x=A1, y=A2, z=Jn)
    for fraction, opacity in zip(RING_LEVELS, RING_OPACITIES):
        for contour in contours.lines(j_min + fraction * (j_max - j_min)):
            image, _ = project(contour[:, 0], contour[:, 1], np.full(len(contour), fraction))
            rings.append(line(image, theme["minimum_halo"], 1.1, opacity))

    centre, scale = project(minimum[0].item(), minimum[1].item(), 0.0)
    scale = float(scale)

    # The mesh fades with the distance from the minimum (an ellipse, wider than tall)
    a, b, c, d = theme["falloff"]
    defs.append(
        f'<radialGradient id="light_falloff_fill" gradientUnits="userSpaceOnUse" cx="{centre[0]:.1f}" '
        f'cy="{centre[1]:.1f}" r="{FALLOFF_RADIUS}" '
        f'gradientTransform="translate(0 {centre[1] * (1 - FALLOFF_FLATTEN):.1f}) scale(1 {FALLOFF_FLATTEN})">'
        + stops([(0, theme["ground_edge"], a), (0.22, theme["ground_edge"], b),
                 (0.5, theme["ground_edge"], c), (1, theme["ground_edge"], d)])
        + "</radialGradient>")
    defs.append(
        f'<radialGradient id="ground_fill" gradientUnits="userSpaceOnUse" cx="{centre[0]:.1f}" '
        f'cy="{centre[1]:.1f}" r="{GROUND_RADIUS}">'
        + stops([(0, theme["ground_centre"], 1), (1, theme["ground_edge"], 1)]) + "</radialGradient>")
    halo = theme["minimum_halo"]
    defs.append(radial_gradient("minimum_halo_fill", [(0, halo, 0.75 * theme["glow"]),
                                                      (0.35, halo, 0.22 * theme["glow"]), (1, halo, 0)]))

    print(f"state: trajectory {TRIAL}, t = {float(sim['trial_time'][STATE_STEP]):.3f} s, "
          f"q = {[round(v, 3) for v in q.tolist()]}, dq = {[round(v, 3) for v in dq.tolist()]}")
    print(f"minimum of the objective: {[round(v, 3) for v in minimum.tolist()]} rad/s^2; "
          f"acceleration stored by the simulation: "
          f"{[round(v, 3) for v in sim['all_pred_ddq'][TRIAL, STATE_STEP].tolist()]} rad/s^2")
    print(f"curvatures at the minimum: {[round(float(v), 3) for v in curvature]}; "
          f"valley direction: {[round(float(v), 3) for v in valley]}")

    return [
        group("background", [f'<rect width="{WIDTH}" height="{HEIGHT}" fill="url(#ground_fill)"/>']),
        group("objective_mesh", [group("mesh_lines_along_acceleration_1", lines_1),
                                 group("mesh_lines_along_acceleration_2", lines_2)]),
        group("objective_level_sets", rings),
        group("light_falloff", [f'<rect width="{WIDTH}" height="{HEIGHT}" fill="url(#light_falloff_fill)"/>']),
        group("minimum", [disc(centre[0], centre[1], 58 * scale, "url(#minimum_halo_fill)", "minimum_halo"),
                          disc(centre[0], centre[1], 8.5 * scale, theme["minimum"], "minimum_disc")]),
    ]


# -------------------------------------------------------------------- pendulum
def pendulum_layers(sim, theme, defs):
    pivot = np.array(PIVOT)
    blue, red = theme["blue"], theme["red"]
    blue_core, red_core, glow = theme["blue_core"], theme["red_core"], theme["glow"]

    q = sim["all_pred_q"][TRIAL].numpy().astype(float)
    mass_1 = pivot + METRE * L1 * np.stack((np.sin(q[:, 0]), np.cos(q[:, 0])), axis=1)
    mass_2 = mass_1 + METRE * L2 * np.stack((np.sin(q[:, 0] + q[:, 1]), np.cos(q[:, 0] + q[:, 1])), axis=1)

    sweep = slice(SWEEP_FIRST, STATE_STEP + 1, 4)
    sweep_1, sweep_2 = mass_1[sweep], mass_2[sweep]

    defs.append(linear_gradient("swept_area_1_fill", sweep_1[0], sweep_1[-1], [(0, blue, 0.0), (1, blue, 0.26 * glow)]))
    defs.append(linear_gradient("swept_area_2_fill", sweep_2[0], sweep_2[-1], [(0, red, 0.0), (1, red, 0.26 * glow)]))
    # A veil in the background colour dims the mesh behind the swept areas
    ground, veil = theme["ground_centre"], theme["veil"]
    defs.append(linear_gradient("veil_fill", sweep_2[0], sweep_2[-1], [(0, ground, 0.0), (0.6, ground, veil), (1, ground, veil)]))
    outline = np.vstack((pivot[None], sweep_1[:1], sweep_2, sweep_1[-1:]))
    swept = group("swept_areas", [
        shape(outline, "url(#veil_fill)", name="veil_behind_the_pendulum"),
        shape(np.vstack((pivot[None], sweep_1)), "url(#swept_area_1_fill)", name="swept_area_link_1"),
        shape(np.vstack((sweep_1, sweep_2[::-1])), "url(#swept_area_2_fill)", name="swept_area_link_2"),
    ])

    paths = group("mass_paths_5_s", [line(mass_1[::2], blue, 1.0, 0.32), line(mass_2[::2], red, 1.0, 0.32)])

    steps = list(range(SWEEP_FIRST, STATE_STEP, POSE_EVERY))
    poses = []
    for k, step in enumerate(steps):
        a, b = mass_1[step], mass_2[step]
        poses.append(group(f"pose_{k + 1:02d}", [
            line([pivot, a], blue, 1.8), line([a, b], red, 1.8),
            disc(a[0], a[1], 3.6, blue), disc(b[0], b[1], 3.6, red),
        ], opacity=0.10 + 0.50 * (k / (len(steps) - 1)) ** 1.6))

    trails = group("mass_trails", [
        shape(tapered_trail(sweep_1, 22, 1.2), blue, 0.10 * glow), shape(tapered_trail(sweep_1, 7.5), blue_core, 0.85),
        shape(tapered_trail(sweep_2, 22, 1.2), red, 0.10 * glow), shape(tapered_trail(sweep_2, 7.5), red_core, 0.85),
    ])

    a, b = mass_1[STATE_STEP], mass_2[STATE_STEP]
    defs.append(radial_gradient("mass_1_fill", [(0, "#FFFFFF", 1), (0.45, blue_core, 1), (1, blue, 1)], 0.36, 0.34, 0.75))
    defs.append(radial_gradient("mass_2_fill", [(0, "#FFFFFF", 1), (0.45, red_core, 1), (1, red, 1)], 0.36, 0.34, 0.75))
    defs.append(radial_gradient("mass_1_halo_fill", [(0, blue, 0.55 * glow), (1, blue, 0)]))
    defs.append(radial_gradient("mass_2_halo_fill", [(0, red, 0.55 * glow), (1, red, 0)]))
    bright = group("bright_pose", [
        group("link_1", glowing_line([pivot, a], blue_core, blue, 4.2, glow)),
        group("link_2", glowing_line([a, b], red_core, red, 4.2, glow)),
        disc(a[0], a[1], 40, "url(#mass_1_halo_fill)", "mass_1_halo"),
        disc(b[0], b[1], 40, "url(#mass_2_halo_fill)", "mass_2_halo"),
        disc(a[0], a[1], 12.5, "url(#mass_1_fill)", "mass_1"),
        disc(b[0], b[1], 12.5, "url(#mass_2_fill)", "mass_2"),
    ])
    hinge = group("pivot", [disc(pivot[0], pivot[1], 6.5, theme["pivot"]),
                            disc(pivot[0], pivot[1], 3.0, theme["ground_edge"])])

    return [group("pendulum", [swept, paths, group("earlier_poses", poses), trails, bright, hinge])]


def compose(sim, stats, model, theme):
    defs = []
    layers = objective_layers(sim, stats, model, theme, defs) + pendulum_layers(sim, theme, defs)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
            f'viewBox="0 0 {WIDTH} {HEIGHT}">\n<defs>\n' + "\n".join(defs) + "\n</defs>\n"
            + "\n".join(layers) + "\n</svg>\n")


def main():
    sim, stats, model = load()
    for name, theme in THEMES.items():
        text = compose(sim, stats, model, theme)
        file_name = f"Featured Image - {name} background.svg"
        with open(os.path.join(HERE, file_name), "w") as handle:
            handle.write(text)
        print("wrote", file_name, f"({len(text) / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
