from matplotlib.widgets import Slider
import numpy as np
import scipy.io
import matplotlib
import matplotlib.pyplot as plt

import os
import scipy.interpolate

matplotlib.use("TkAgg")
INPUT_DIR = "../data/pre_interpolation/lbbb"


def unwrap(x):
    while isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        x = x[0]
    return x

def load_case(case_name):
    beat_file = os.path.join(INPUT_DIR, f"{case_name}_beats")
    torso_file = os.path.join(INPUT_DIR, f"{case_name}_torso")

    torso = scipy.io.loadmat(torso_file)
    beats = scipy.io.loadmat(beat_file)

    verts2d = np.array(unwrap(torso["xy"]))
    faces = np.array(unwrap(torso["faces"])) - 1
    verts3d = np.array(unwrap(torso["XYZ"]))

    V = np.array(unwrap(beats["bodyPots"]))  # (N, T)

    print("verts2d:", verts2d.shape)
    print("verts3d:", verts3d.shape)
    print("V:", V.shape)

    # ------------------------------------------------------------
    # sanity check
    # ------------------------------------------------------------
    if V.shape[0] != verts2d.shape[0]:
        raise ValueError(
            f"Mismatch: verts2d={verts2d.shape[0]} vs V={V.shape[0]}"
        )

    # ------------------------------------------------------------
    # define CLEAN values (THIS WAS MISSING BEFORE)
    # ------------------------------------------------------------
    values = V

    # remove rows where ALL timesteps are NaN
    valid_mask = ~np.isnan(values).all(axis=1)

    # ------------------------------------------------------------
    # rebuild vertex mapping
    # ------------------------------------------------------------
    old_to_new = -np.ones(len(valid_mask), dtype=int)
    old_to_new[np.where(valid_mask)[0]] = np.arange(np.sum(valid_mask))

    verts2d = verts2d[valid_mask]
    verts3d = verts3d[valid_mask]
    values = values[valid_mask]

    # ------------------------------------------------------------
    # rebuild faces safely
    # ------------------------------------------------------------
    faces_new = old_to_new[faces]

    faces = faces_new[np.all(faces_new >= 0, axis=1)]

    return verts2d, faces, verts3d, values

def interactive_3d_view(verts3d, values):
    """
    verts3d: (N, 3)
    values: (N, T)
    """

    T = values.shape[1]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    t0 = 0
    v = values[:, t0]

    sc = ax.scatter(
        verts3d[:, 0],
        verts3d[:, 1],
        verts3d[:, 2],
        c=v,
        cmap='turbo',
        s=20
    )

    cb = plt.colorbar(sc, ax=ax, shrink=0.6)
    cb.set_label("Potential")

    ax.set_title(f"Time step: {t0}")

    # ============================================================
    # SLIDER AXIS
    # ============================================================
    slider_ax = plt.axes([0.2, 0.02, 0.6, 0.03])
    slider = Slider(slider_ax, "Time", 0, T - 1, valinit=t0, valstep=1)

    # ============================================================
    # UPDATE FUNCTION
    # ============================================================
    def update(val):
        t = int(slider.val)

        ax.clear()

        v = values[:, t]

        sc = ax.scatter(
            verts3d[:, 0],
            verts3d[:, 1],
            verts3d[:, 2],
            c=v,
            cmap='turbo',
            s=20
        )

        ax.set_title(f"Time step: {t}")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show(block=True)

verts2d, faces, verts3d, values = load_case("ECGiCRT3")

interactive_3d_view(verts3d, values)