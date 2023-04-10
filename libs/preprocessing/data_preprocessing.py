from global_features import Global_features
from task_preprocessing import Preprocess_datasets
from image_sizes import Calculate_sizes

data_root = '/media/SSD0/nfvalderrama/Vessel_Segmentation/data/datasets/OASIS3/vessels_segmentation'
out_directory = '/media/SSD0/nfvalderrama/Vessel_Segmentation/data/Vessel_Segmentation'
num_workers = 50
process_again=True

# Calculate the statistics of the original dataset
Global_features(data_root, num_workers)

# # Preprocess the datasets
Preprocess_datasets(out_directory, data_root, num_workers, remake=process_again)

# # Calculate the dataset sizes (used to plan the experiments)
Calculate_sizes(out_directory, num_workers, remake=process_again)

