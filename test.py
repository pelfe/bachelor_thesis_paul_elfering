from deep_learning_based_estimation_of_heart_surface_potentials_main import *
from deep_learning_based_estimation_of_heart_surface_potentials_main.create_data import get_data
from deep_learning_based_estimation_of_heart_surface_potentials_main.test_model import get_mse_data, test_model
import tensorflow as tf

# 0 define parameters
img_dim = 32
flattened = False
epochs = 10



# 1 load model and data
model = tf.saved_model.load(r"C:\Users\paule\Desktop\Data Science and AI\Year 3\Bachelor Thesis\Data\original_tiantian_models\plots & inception\inception_noskip\ivf_ct\final_model")
#datasets =
datasets = [[1, 6, 26, 55, 56, 57], # Sinus beats
            [3, 7, 15, 16, 17, 18, 29, 36, 37, 38, 67, 68, 69, 70, 71], # LVLat
            [2, 10, 11, 12, 13 ,14, 28, 31, 32, 33, 34, 35, 63, 64, 65, 66], # LVant
            [4, 8, 19, 20, 39, 40, 41, 42, 43, 44, 72, 73, 74, 75, 76, 77, 79], # LVpost
            [5, 9, 30, 45, 46, 47, 48, 49, 80, 81, 82, 83, 84, 85, 86, 87, 88], # RVlat
            [15, 36, 68, 10, 33, 64, 41, 73, 45, 80], # NEW
            [15, 36, 67, 10, 31, 63, 39, 72, 45, 80],# OLD
            [67, 68, 69, 70, 71, 63, 64, 65, 66, 72, 73, 74, 75, 76, 77, 79, 89, 90, 91, 80, 81, 82, 83, 84, 85, 86, 87, 88, 58, 59, 60, 61, 62, 78, 92, 93] # Train dog 123
            ]

# 2. Get test data again (same parameters as training)
(train_x, train_y), (test_X, test_y) = get_data(img_dim, datasets[0], flattened)

# 3. Calculate MSE on test data
test_mse = get_mse_data(model, test_X, test_y)
print(f"Test MSE: {test_mse}")

# 4. Visualize predictions
test_model(model, test_X, test_y)