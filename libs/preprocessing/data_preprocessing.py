from global_features import Global_features
from task_preprocessing import Preprocess_datasets
from image_sizes import Calculate_sizes

data_root = '../../datasets/OASIS-3/original'
out_directory = 'your/saving/processed/data/path'
num_workers = 50
# Change to false if want to avoid the processing on data that has been 
# already processed.
process_again=True

# Calculate the statistics of the original dataset
Global_features(data_root, num_workers)

# # Preprocess the datasets
Preprocess_datasets(out_directory, data_root, num_workers, remake=process_again)

# # Calculate the dataset sizes (used to plan the experiments)
Calculate_sizes(out_directory, num_workers, remake=process_again)

