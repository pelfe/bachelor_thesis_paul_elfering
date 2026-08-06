
from deep_learning_based_estimation_of_heart_surface_potentials_main import data_functions_paul


### this is to make the images into npy files
stack_size = 11
save_folder = "data/post_interpolation/healthy/npy_stacked_" + str(stack_size)
load_folder_heart = "data/post_interpolation/healthy/heart_black_white"
load_folder_torso = "data/post_interpolation/healthy/torso_black_white"
#subjects = ["B6", "ECGiCRT2", "ECGiCRT3", "ECGiCRT4", "ECGiCRT10", "ECGiCRT17", "L79"]
#subjects = ["A91", "A95", "B62", "F4", "F5", "G39", "H19", "I60", "J74", "K72", "L8", "M7", "M58", "P12", "Q55", "R9", "S7", "T12", "W2", "W5", "W18", "W93", "X2", "X35", "Y46", "Y82", "Z3", "Z34"]
#subjects = ["ct1", "ct2", "ct3", "ct4", "ct5", "ct6", "ct7", "ct8", "ct9", "ct10", "ct11"]
subjects = [ "ct1"]

for subject in subjects:
    print(subject)
    data_functions_paul.images_into_npy(load_folder_heart, load_folder_torso, save_folder, subject, stack_size)
    data_functions_paul.images_into_npy_entropy(load_folder_heart, load_folder_torso, save_folder, subject, stack_size)