from utils import DataModule
import numpy as np
import sys

DATA_DIR_PATH = "D:/AI/data/english accent classification/"
CUT_SEC = 10
try:
    N_MFCC = int(sys.argv[1])
except IndexError:
    print(f"sys.argv[1] is empty. please check the argument.")
    exit()
SR = 48000

dm = DataModule(data_dir_path=DATA_DIR_PATH, training_dir_name="train", test_dir_name="test")

# for training data
features, norm_features, labels = dm.training_data_preprocessing(cut_sec=CUT_SEC, n_mfcc=N_MFCC, sr=SR)

print(np.shape(features), np.shape(norm_features), np.shape(labels))
np.savez_compressed(file=f"npz/training_dataset_cut-sec-{CUT_SEC}_n-mfcc-{N_MFCC}_sr-{SR}.npz",
                    features=features, norm_features=norm_features, labels=labels)
