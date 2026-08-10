
from deep_learning_estimation_paul import visualize_model_functions as visualize_model_history

#model_name = "mixed_15_stacks_gan"
#history_path =
history = visualize_model_history.load_history(
    "data/models/pauls_models/wrong_params/mixed_2x2patch_15stacks_gan/history.json")

# Overview
visualize_model_history.plot_training_overview(history)
