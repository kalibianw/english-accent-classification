from utils import DataModule
import numpy as np

DATA_DIR_PATH = "D:/AI/data/english accent classification/"

# for training data
dm = DataModule(data_dir_path=DATA_DIR_PATH, training_dir_name="train", test_dir_name="test")
features, norm_features, labels = dm.training_data_preprocessing(10, 80)

print(np.shape(features), np.shape(norm_features), np.shape(labels))
np.savez_compressed(file="npz/training_dataset.npz", features=features, norm_features=norm_features, labels=labels)
