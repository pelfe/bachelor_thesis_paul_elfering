import os
import numpy as np
import scipy.io
import matplotlib

# ----------------------------
# FORCE INTERACTIVE BACKEND
# ----------------------------
matplotlib.use("Qt5Agg")  # fallback: "TkAgg"

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Slider

# --------------------------------------------------------
# PATHS
# --------------------------------------------------------
input_path = "../data/pre_interpolation/lbbb"
filename_bullseye = "B6_bullseye.mat"
filename_beats = "B6_beats.mat"

img_dir = "../data/post_interpolation/lbbb/B6/heart"

# --------------------------------------------------------
# LOAD MATLAB DATA
# --------------------------------------------------------
print("Loading MATLAB data...")

bullseye = scipy.io.loadmat(os.path.join(input_path, filename_bullseye))
beats = scipy.io.loadmat(os.path.join(input_path, filename_beats))

bullseye_vertices = np.array(bullseye["xy"])
map_array = np.array(bullseye["map"])

V = beats["potsTikhonov"]

# ensure correct orientation
if V.shape[0] < V.shape[1]:
    V = V.T

values = V[map_array[:, 1] - 1, :]   # (nodes, time)

print("raw values:", values.shape)

# --------------------------------------------------------
# NORMALIZE TO [0,1]
# --------------------------------------------------------
values = (values - values.min()) / (values.max() - values.min() + 1e-8)

# --------------------------------------------------------
# LOAD IMAGES
# --------------------------------------------------------
print("Loading images...")

img_files = sorted([
    f for f in os.listdir(img_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

if len(img_files) == 0:
    raise RuntimeError("No images found!")

images = []
for f in img_files:
    img = mpimg.imread(os.path.join(img_dir, f))

    if img.ndim == 3:
        img = img[:, :, 0]

    images.append(img)

images = np.array(images)

print("images:", images.shape)

# --------------------------------------------------------
# ALIGN TIME AXIS
# --------------------------------------------------------
T = min(values.shape[1], images.shape[0])

values = values[:, :T]
images = images[:T]

print("final T:", T)

# --------------------------------------------------------
# NODE COORDINATES
# --------------------------------------------------------
x = bullseye_vertices[:, 0]
y = bullseye_vertices[:, 1]

print("GLOBAL min:", values.min())
print("GLOBAL max:", values.max())
print("GLOBAL std:", values.std())
# --------------------------------------------------------
# PLOT SETUP
# --------------------------------------------------------
t0 = 0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plt.subplots_adjust(bottom=0.2)

# LEFT: cardiac nodes
sc = ax1.scatter(
    x, y,
    c=values[:, t0],
    cmap="jet",
    s=20,
    vmin=0,
    vmax=1
)

ax1.set_title("Bullseye (0–1 normalized)")
ax1.set_aspect("equal")
plt.colorbar(sc, ax=ax1)

# RIGHT: image
img_disp = ax2.imshow(images[t0], cmap="jet")
ax2.set_title("2D heart image")
plt.colorbar(img_disp, ax=ax2)

# --------------------------------------------------------
# SLIDER
# --------------------------------------------------------
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, "Time", 0, T - 1, valinit=t0, valstep=1)

def update(val):
    t = int(slider.val)

    colors = values[:, t]

    sc.set_array(colors)
    sc.set_clim(0, 1)   # IMPORTANT: re-lock scale

    img_disp.set_data(images[t])

    fig.canvas.draw_idle()

slider.on_changed(update)

# --------------------------------------------------------
# START GUI LOOP (IMPORTANT)
# --------------------------------------------------------
plt.show(block=True)