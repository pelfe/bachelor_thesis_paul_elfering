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
    bull_file = os.path.join(INPUT_DIR, f"{case_name}_bullseye")

    bullseye = scipy.io.loadmat(bull_file)
    beats = scipy.io.loadmat(beat_file)

    vertices2d = np.array(unwrap(bullseye["xy"]))
    faces = np.array(unwrap(bullseye["facesxy"])) - 1
    vertices3d = np.array(unwrap(bullseye["XY3D"]))
    map_array = np.array(unwrap(bullseye["map"]))

    V_raw = beats["potsTikhonov"]
    V = np.array(unwrap(V_raw))

    print("CASE:", case_name, flush=True)
    print("V shape:", np.shape(V), flush=True)
    print("map shape:", np.shape(map_array), flush=True)

    # SAFE indexing only if valid
    if map_array.ndim == 2:
        values = V[map_array[:, 1] - 1, :]
    else:
        raise ValueError(f"map_array invalid shape: {map_array.shape}")

    return vertices2d, faces, vertices3d, values

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

verts2d, faces, verts3d, values = load_case("B6")

interactive_3d_view(verts3d, values)