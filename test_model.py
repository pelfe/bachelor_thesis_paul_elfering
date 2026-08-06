import os

os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
os.environ["TF_USE_CUDNN_FRONTEND"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
tf.config.optimizer.set_jit(False)
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)
import deep_learning_estimation_paul.test_functions_paul as test_functions_paul


#model_path = "data/models/pauls_models/mixed_15stacks_gan/generator.keras"
#gan_model = tf.keras.models.load_model(model_path)

#save_path = "data/output/mixed_15stacks_gan_lbbb_metrics.csv"
#data_path = "data/post_interpolation/lbbb/npy_stacked_15/size64_subject"
#subjects = ["L79", "ECGiCRT17"]
#test_functions_paul.create_metric_summary(subjects, data_path ,save_path, gan_model)

#save_path = "data/output/mixed_15stacks_gan_idiohealthy_metrics.csv"
#data_path = [
#    "data/post_interpolation/healthy/npy_stacked_15/size64_subject",
#    "data/post_interpolation/idiopathic/npy_stacked_15/size64_subject",]
#subjects = [
#    ["ct9", "ct10", "ct11"],
#    ["X35", "Y46", "Y82", "Z3", "Z34"],]
#test_functions_paul.create_metric_summary(subjects, data_path ,save_path, gan_model)


generator1_path = "data/models/pauls_models/mixed_15stacks_gan_original_high_lr/generator.keras"
discriminator1_path = "data/models/pauls_models/mixed_15stacks_gan_original_high_lr/discriminator.keras"
generator_model1 = tf.keras.models.load_model(generator1_path)
discriminator_model1 = tf.keras.models.load_model(discriminator1_path)

#model2_path = "data/models/pauls_models/mixed_2x2_15stacks_ac_gan/generator.keras"
#discriminator2_path = "data/models/pauls_models/mixed_2x2_15stacks_ac_gan/generator.keras"
#generator_model2 = tf.keras.models.load_model(model2_path)
#discriminator_model2 = tf.keras.models.load_model(discriminator1_path)


subject_path = "data/post_interpolation/lbbb/npy_stacked_15/size64_subject"
subject = "ECGiCRT17"
test_functions_paul.visualize_input_output_truth_error(
    [generator_model1],
    [discriminator_model1],
    subject,
    subject_path
   )