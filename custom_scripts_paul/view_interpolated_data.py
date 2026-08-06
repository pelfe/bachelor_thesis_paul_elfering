from matplotlib.widgets import Slider
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

BASE_DIR = "../data/post_interpolation/lbbb"

CASE = "ECGiCRT4"


def interactive_view(case_name):

    heart_dir = os.path.join(BASE_DIR, f"{case_name}_heart")
    torso_dir = os.path.join(BASE_DIR, f"{case_name}_torso")

    heart_files = sorted(
        [f for f in os.listdir(heart_dir) if f.endswith(".png")]
    )

    torso_files = sorted(
        [f for f in os.listdir(torso_dir) if f.endswith(".png")]
    )

    num_frames = min(len(heart_files), len(torso_files))

    # ============================================================
    # LOAD FIRST FRAME
    # ============================================================

    heart_img = mpimg.imread(
        os.path.join(heart_dir, heart_files[0])
    )

    torso_img = mpimg.imread(
        os.path.join(torso_dir, torso_files[0])
    )

    print("Heart shape:", heart_img.shape)
    print("Torso shape:", torso_img.shape)

    # Convert RGBA/RGB -> grayscale channel
    if heart_img.ndim == 3:
        heart_img = heart_img[..., 0]

    if torso_img.ndim == 3:
        torso_img = torso_img[..., 0]

    # ============================================================
    # FIGURE
    # ============================================================

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    plt.subplots_adjust(bottom=0.15)

    im1 = ax1.imshow(
        heart_img,
        cmap="jet",
        origin="lower"
    )

    ax1.set_title("Heart")
    ax1.axis("off")

    im2 = ax2.imshow(
        torso_img,
        cmap="jet",
        origin="lower"
    )

    ax2.set_title("Torso")
    ax2.axis("off")

    # colorbars
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    fig.suptitle("Time step 0")

    # ============================================================
    # SLIDER
    # ============================================================

    slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])

    slider = Slider(
        slider_ax,
        "Time",
        0,
        num_frames - 1,
        valinit=0,
        valstep=1
    )

    # ============================================================
    # UPDATE
    # ============================================================

    def update(val):

        t = int(slider.val)

        heart_img = mpimg.imread(
            os.path.join(heart_dir, heart_files[t])
        )

        torso_img = mpimg.imread(
            os.path.join(torso_dir, torso_files[t])
        )

        if heart_img.ndim == 3:
            heart_img = heart_img[..., 0]

        if torso_img.ndim == 3:
            torso_img = torso_img[..., 0]

        im1.set_data(heart_img)
        im2.set_data(torso_img)

        # update color scaling
        im1.set_clim(
            vmin=heart_img.min(),
            vmax=heart_img.max()
        )

        im2.set_clim(
            vmin=torso_img.min(),
            vmax=torso_img.max()
        )

        fig.suptitle(f"Time step {t}")

        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show(block=True)


if __name__ == "__main__":
    interactive_view(CASE)