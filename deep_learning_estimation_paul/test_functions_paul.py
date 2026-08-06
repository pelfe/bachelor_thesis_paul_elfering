
import tensorflow as tf
from deep_learning_estimation_paul.data_functions_paul import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# make sure Qt backend is used
import matplotlib
matplotlib.use("Qt5Agg")  # important for PyCharm interactive windows
from matplotlib.widgets import Slider

def inspect_saved_model(model):
    print("=== Available Signatures ===")
    print(list(model.signatures.keys()))

    for name, fn in model.signatures.items():
        print(f"\n=== Signature: {name} ===")

        print("Inputs:")
        for tensor_name, spec in fn.structured_input_signature[1].items():
            print(
                f"  {tensor_name}: "
                f"shape={spec.shape}, dtype={spec.dtype}"
            )

        print("Outputs:")
        for tensor_name, spec in fn.structured_outputs.items():
            print(
                f"  {tensor_name}: "
                f"shape={spec.shape}, dtype={spec.dtype}"
            )

def get_full_prediction(model, x):
    predicted = model.predict(x, batch_size=32)
    return predicted

def get_middle_prediction(model, x, middle=7):
    #creates the average of a frame prediction based on all time-windows where it was predicted
    predicted = model.predict(x, batch_size=32)
    print(predicted.shape)
    predicted = predicted[:,:,:, middle]
    #predicted = predicted[5:-5]  # Remove first and last 5 elements along axis 0
    #predicted = tf.convert_to_tensor(predicted, dtype=tf.float32)
    return predicted

def get_ae_per_pixel(y_true, y_pred):
    abs_error = np.abs(y_pred - y_true)
    return abs_error

def get_se(y_true, y_pred):
    se_per_frame = np.square(np.subtract(y_pred, y_true))
    return se_per_frame

def get_mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

def get_mae_std(y_true, y_pred):
    frame_mae = np.mean(np.abs(y_pred-y_true), axis=(1,2))
    return frame_mae.std()

def get_ae(y_true, y_pred):
    return np.mean(np.abs(y_pred-y_true), axis=(1,2))

def get_mse(y_true, y_pred):
    return np.mean((y_pred-y_true)**2)

def get_mse_std(y_true, y_pred):
    frame_mse = np.mean((y_pred - y_true)**2, axis=(1,2))
    return frame_mse.std()

def get_ssim_per_frame(y_true, y_pred):
    y_true = y_true[...,None]
    y_pred = y_pred[...,None]
    return tf.image.ssim(y_true,y_pred,max_val=1.0,filter_size=6).numpy()

def SSIMLoss(y_true,y_pred):
    return np.mean(get_ssim_per_frame(y_true,y_pred))

def get_ssim_std(y_true,y_pred):
    return get_ssim_per_frame(y_true,y_pred).std()


def get_metrics_for_subject(model, x, y):
    # yes this function is calculating the raw mse, mae, ssim values 3 times, this could be a lot more efficient but this only needs to be fixed if the computational strain is actually impactful
    predicted = get_middle_prediction(model, x)
    y = y[:, :, :, 0]

    mse = get_mse(y, predicted)
    mae = get_mae(y, predicted)
    ssim = SSIMLoss(y, predicted)
    mae_std = get_mae_std(y, predicted)
    mse_std = get_mse_std(y, predicted)
    ssim_std = get_ssim_std(y, predicted)
    se_vals = get_se(y, predicted)
    ae_vals = get_ae(y, predicted)
    not_mean_ssim_vals = get_ssim_per_frame(y, predicted)

    print("MSE: " + str(mse) + "STD: " + str(mse_std))
    print("MAE: " + str(mae) + "STD: " + str(mae_std))
    print("SSIM: " + str(ssim) + "STD: " + str(ssim_std))
    return mse, mae, ssim, mse_std, mae_std, ssim_std, se_vals, ae_vals, not_mean_ssim_vals

def create_metric_summary(subjects, load_path, save_path, model):
    if isinstance(load_path, (list, tuple)):
        if not isinstance(subjects[0], (list, tuple)):
            raise ValueError("When load_path is a list, subjects must be a nested list.")
        subject_groups = zip(load_path, subjects)
    else:
        subject_groups = [(load_path, subjects)]

    rows = []
    total_mse_sum = 0
    total_mae_sum = 0
    total_ssim_sum = 0
    total_count = 0
    all_mse = []
    all_mae = []
    all_ssim = []

    for current_load_path, current_subjects in subject_groups:
        for subject in current_subjects:
            print("subject:", subject)

            subject_path = current_load_path + subject
            test_x, test_y = load_data(subject_path)
            test_x = tf.cast(test_x, tf.float32)
            test_y = tf.cast(test_y, tf.float32)

            (subject_mse, subject_mae, subject_ssim,
             subject_mse_std, subject_mae_std, subject_ssim_std,
             se_vals, ae_vals, not_mean_ssim_vals) = get_metrics_for_subject(
                model, test_x, test_y
            )

            n = len(test_x)
            total_mse_sum += subject_mse * n
            total_mae_sum += subject_mae * n
            total_ssim_sum += subject_ssim * n
            total_count += n

            all_mse.append(se_vals)
            all_mae.append(ae_vals)
            all_ssim.append(not_mean_ssim_vals)

            rows.append({
                "subject": subject,
                "n_samples": n,
                "mse_mean": subject_mse,
                "mse_std": subject_mse_std,
                "mae_mean": subject_mae,
                "mae_std": subject_mae_std,
                "ssim_mean": subject_ssim,
                "ssim_std": subject_ssim_std
            })

    # calculate global stats
    final_mse = total_mse_sum / total_count
    final_mae = total_mae_sum / total_count
    final_ssim = total_ssim_sum / total_count

    all_mse = np.concatenate(all_mse)
    all_mae = np.concatenate(all_mae)
    all_ssim = np.concatenate(all_ssim)

    global_mse_std = np.std(all_mse)
    global_mae_std = np.std(all_mae)
    global_ssim_std = np.std(all_ssim)

    df = pd.DataFrame(rows)

    print("\nFINAL RESULTS (weighted by samples)")
    print("MSE:", final_mse, "std:", global_mse_std)
    print("MAE:", final_mae, "std:", global_mae_std)
    print("SSIM:", final_ssim, "std:", global_ssim_std)

    df_total = pd.DataFrame([{
        "subject": "TOTAL",
        "n_samples": total_count,
        "mse_mean": final_mse,
        "mse_std": global_mse_std,
        "mae_mean": final_mae,
        "mae_std": global_mae_std,
        "ssim_mean": final_ssim,
        "ssim_std": global_ssim_std
    }])

    df = pd.concat([df, df_total], ignore_index=True)

    df.to_csv(save_path, index=False)
    print("Saved to:", save_path)

def visualize_all_corresponding_outputs(model, subject, load_path):
    subject_path = load_path + subject
    test_x, test_y = load_data(subject_path)
    test_x = tf.cast(test_x, tf.float32)
    test_y = tf.cast(test_y, tf.float32)
    predicted = get_full_prediction(model, test_x)

    output_data = predicted[:,:,:,10]
    ground_truth_data = test_y[:,:,:,10]

    N = test_x.shape[0]

    # -------------------------
    # FIGURE (2 IMAGES)
    # -------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    plt.subplots_adjust(bottom=0.15)

    # Output
    img_out = ax1.imshow(output_data[0], cmap="jet", vmin=0, vmax=1)
    ax1.set_title("Output")
    ax1.axis("off")

    # Ground truth
    img_gt = ax2.imshow(ground_truth_data[0], cmap="jet", vmin=0, vmax=1)
    ax2.set_title("Ground Truth")
    ax2.axis("off")

    # -------------------------
    # SLIDER
    # -------------------------
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, "t", 0, N - 1, valinit=0, valstep=1)

    # -------------------------
    # UPDATE FUNCTION
    # -------------------------
    def update(val):
        t = int(slider.val)

        img_out.set_data(output_data[t])
        img_gt.set_data(ground_truth_data[t])

        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()

def visualize_all_in_out(model, subject, load_path):

    subject_path = load_path + subject
    test_x, test_y = load_data(subject_path)

    test_x = tf.cast(test_x, tf.float32)
    test_y = tf.cast(test_y, tf.float32)

    predicted = get_full_prediction(model, test_x)

    N = test_x.shape[0]
    C = test_x.shape[-1]  # 11 channels

    # -------------------------
    # FIGURE (3 x 11 GRID)
    # -------------------------
    fig, axes = plt.subplots(3, C, figsize=(2 * C, 7))
    plt.subplots_adjust(bottom=0.15)

    imgs_in = []
    imgs_out = []
    imgs_gt = []

    # init timestep 0
    for c in range(C):

        im_in = axes[0, c].imshow(test_x[0, :, :, c], cmap="jet", vmin=0, vmax=1)
        axes[0, c].set_title(f"In {c}")
        axes[0, c].axis("off")
        imgs_in.append(im_in)

        im_out = axes[1, c].imshow(predicted[0, :, :, c], cmap="jet", vmin=0, vmax=1)
        axes[1, c].set_title(f"Out {c}")
        axes[1, c].axis("off")
        imgs_out.append(im_out)

        im_gt = axes[2, c].imshow(test_y[0, :, :, c], cmap="jet", vmin=0, vmax=1)
        axes[2, c].set_title(f"GT {c}")
        axes[2, c].axis("off")
        imgs_gt.append(im_gt)

    # -------------------------
    # SLIDER
    # -------------------------
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, "t", 0, N - 1, valinit=0, valstep=1)

    # -------------------------
    # UPDATE FUNCTION
    # -------------------------
    def update(val):
        t = int(slider.val)

        for c in range(C):
            imgs_in[c].set_data(test_x[t, :, :, c])
            imgs_out[c].set_data(predicted[t, :, :, c])
            imgs_gt[c].set_data(test_y[t, :, :, c])

        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()




def visualize_input_output_truth_error(models, discriminators, subject, load_path):

    # -------------------------
    # Allow single model or list
    # -------------------------

    if not isinstance(models, (list, tuple)):
        models = [models]

    if not isinstance(discriminators, (list, tuple)):
        discriminators = [discriminators]

    assert len(models) == len(discriminators)


    subject_path = load_path + subject
    test_x, test_y = load_data(subject_path)

    test_x = tf.cast(test_x, tf.float32)
    test_y = tf.cast(test_y, tf.float32)
    # Keep full temporal stack for discriminator
    test_x_disc = test_x.numpy()
    test_y_disc = test_y.numpy()

    input_data = test_x[:, :, :, 0].numpy()
    ground_truth_data = test_y[:, :, :, 0].numpy()

    # Predictions
    pred_list = []
    error_list = []

    discriminator_real_scores = []
    discriminator_fake_scores = []
    for i, model in enumerate(models):
        pred = get_middle_prediction(model, test_x)
        pred_list.append(pred)
        error_list.append(np.abs(get_ae(ground_truth_data, pred)))
        # -----------------------------
        # Discriminator scores
        # -----------------------------
        all_predictions = get_full_prediction(model, test_x)

        real_score = discriminators[i].predict(
            [test_x_disc, test_y_disc],
            verbose=0
        )

        fake_score = discriminators[i].predict(
            [test_x_disc, all_predictions],
            verbose=0
        )
        print("real min/max:", np.min(real_score), np.max(real_score))
        print("fake min/max:", np.min(fake_score), np.max(fake_score))
        # Handle PatchGAN outputs
        if len(real_score.shape) > 1:
            real_score = np.mean(real_score, axis=tuple(range(1, len(real_score.shape))))

        if len(fake_score.shape) > 1:
            fake_score = np.mean(fake_score, axis=tuple(range(1, len(fake_score.shape))))

        discriminator_real_scores.append(real_score)
        discriminator_fake_scores.append(fake_score)

    N = input_data.shape[0]

    # -------------------------
    # FIGURE LAYOUT
    # -------------------------
    n_models = len(models)
    n_cols = n_models + 2  # Input + models + GT

    fig = plt.figure(figsize=(4 * n_cols, 6))
    gs = fig.add_gridspec(3, n_cols, height_ratios=[2, 1, 1])

    axes = []

    # Input
    ax_in = fig.add_subplot(gs[0, 0])
    axes.append(ax_in)

    # Prediction axes
    pred_axes = []
    for i in range(n_models):
        ax = fig.add_subplot(gs[0, i + 1])
        pred_axes.append(ax)
        axes.append(ax)

    # Ground truth
    ax_gt = fig.add_subplot(gs[0, -1])
    axes.append(ax_gt)

    ax_err = fig.add_subplot(gs[1, :])

    plt.subplots_adjust(bottom=0.15)

    # -------------------------
    # INITIAL IMAGES
    # -------------------------
    im_in = ax_in.imshow(input_data[0], cmap="jet", vmin=0, vmax=1)
    ax_in.set_title("Input")
    ax_in.axis("off")

    pred_images = []
    for i, ax in enumerate(pred_axes):
        im = ax.imshow(pred_list[i][0], cmap="jet", vmin=0, vmax=1)
        ax.set_title(f"Prediction {i+1}")
        ax.axis("off")
        pred_images.append(im)

    im_gt = ax_gt.imshow(ground_truth_data[0], cmap="jet", vmin=0, vmax=1)
    ax_gt.set_title("Ground Truth")
    ax_gt.axis("off")

    # Shared colorbar
    fig.colorbar(im_in, ax=axes, fraction=0.02, pad=0.04)

    # -------------------------
    # ERROR PLOT
    # -------------------------
    markers = []

    for i, err in enumerate(error_list):
        ax_err.plot(err, label=f"Model {i+1}")
        marker, = ax_err.plot(0, err[0], "o")
        markers.append(marker)

    ax_err.set_title("Absolute Error over Time")
    ax_err.set_xlabel("t")
    ax_err.set_ylabel("Error")
    ax_err.legend()

    # -------------------------
    # SLIDER
    # -------------------------
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, "t", 0, N - 1, valinit=0, valstep=1)

    # -------------------------
    # DISCRIMINATOR SCORE PLOT
    # -------------------------

    disc_ax = fig.add_subplot(gs[2, :])

    for i in range(n_models):
        print("real discriminator:", discriminator_real_scores[i])
        print("fake discriminator:", discriminator_fake_scores[i])
        disc_ax.plot(
            discriminator_real_scores[i] * 100,
            label=f"Model {i + 1} Real (%)"
        )

        disc_ax.plot(
            discriminator_fake_scores[i] * 100,
            linestyle="--",
            label=f"Model {i + 1} Fake (%)"
        )

    disc_ax.set_ylim(0, 100)
    disc_ax.set_title("Discriminator confidence (%)")
    disc_ax.set_xlabel("timestep")
    disc_ax.set_ylabel("Probability of being real (%)")
    disc_ax.legend()
    disc_ax.grid(True)

    # -------------------------
    # UPDATE
    # -------------------------
    def update(val):
        t = int(slider.val)

        im_in.set_data(input_data[t])
        im_gt.set_data(ground_truth_data[t])

        for im, pred in zip(pred_images, pred_list):
            im.set_data(pred[t])

        for marker, err in zip(markers, error_list):
            marker.set_data([t], [err[t]])

        disc_text = ""

        for i in range(n_models):
            real_percentage = discriminator_real_scores[i][t] * 100
            fake_percentage = discriminator_fake_scores[i][t] * 100

            disc_text += (
                f"Model {i + 1}: "
                f"Real={real_percentage:.1f}% | "
                f"Fake={fake_percentage:.1f}%\n"
            )

        disc_ax.set_title(
            "Discriminator confidence (%)\n" + disc_text
        )
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()

def visualize_locational_errors(model, subject, load_path):
    subject_path = load_path + subject
    test_x, test_y = load_data(subject_path)
    test_x = tf.cast(test_x, tf.float32)
    test_y = tf.cast(test_y, tf.float32)
    predicted = get_middle_prediction(model, test_x)

    test_x = test_x[:, :, :, 0]
    test_y = test_y[:, :, :, 0]

    input_data = test_x
    ground_truth_data = test_y
    pred_data = predicted
    abs_error = abs(get_ae_per_pixel(ground_truth_data, pred_data))

    # abs_error has shape (N, 64, 64)
    # t = averaging range (number of timesteps on each side)
    t = 100
    N = abs_error.shape[0]

    # --------------------------------------------------
    # Compute averaged error map for EVERY timestep
    # --------------------------------------------------
    # Cumulative sum for fast window averaging
    cumsum = np.concatenate(
        [np.zeros((1, *abs_error.shape[1:])), abs_error.cumsum(axis=0)],
        axis=0
    )

    avg_error_maps = np.empty_like(abs_error)

    for i in range(N):
        # Centered window with clipping at the boundaries
        start = max(i - t, 0)
        end = min(i + t + 1, N)

        avg_error_maps[i] = (cumsum[end] - cumsum[start]) / (end - start)

    # --------------------------------------------------
    # Interactive visualization
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.2)
    print("abs_error shape:", abs_error.shape)
    print("avg_error_maps shape:", avg_error_maps.shape)
    print("avg_error_maps[0] shape:", np.shape(avg_error_maps[0]))
    im = ax.imshow(
        avg_error_maps[0],
        cmap="jet",
        vmin=avg_error_maps.min(),
        vmax=avg_error_maps.max()
    )
    ax.set_title(f"Average Absolute Error Map (t = 0)")
    ax.axis("off")
    fig.colorbar(im, ax=ax)

    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(
        ax_slider,
        "Time",
        0,
        N - 1,
        valinit=0,
        valstep=1
    )

    def update(val):
        i = int(slider.val)
        im.set_data(avg_error_maps[i])
        ax.set_title(f"Average Absolute Error Map (t = {i})")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
