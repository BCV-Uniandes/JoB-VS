from global_features import Global_features
from task_preprocessing import Preprocess_datasets
from image_sizes import Calculate_sizes

num_workers = 10
process_again=True

#### IXI DATASET #####

data_root = '../../datasets/IXI/'
out_directory = 'your/saving/processed/ixi/data/path/'


# # Calculate the statistics of the original dataset
Global_features(data_root, num_workers)

# # Preprocess the datasets
Preprocess_datasets(out_directory, data_root, num_workers, remake=process_again)

# # Calculate the dataset sizes (used to plan the experiments)
Calculate_sizes(out_directory, num_workers, remake=process_again)
    
