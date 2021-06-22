from utils import DataModule
import numpy as np
import sys

DATA_DIR_PATH = "D:/AI/data/english accent classification/"
CUT_SEC = 10
try:
    N_MFCC = int(sys.argv[1])
except IndexError:
    print(f"IndexError. please check the argument.")
    exit()
SR = 48000
SPLITED_CNT = 0

dm = DataModule(data_dir_path=DATA_DIR_PATH, training_dir_name="train", test_dir_name="test")

# for training data
features, norm_features, labels = dm.training_data_preprocessing(cut_sec=CUT_SEC, n_mfcc=N_MFCC, sr=SR, splited=2500)

print(np.shape(features), np.shape(norm_features), np.shape(labels))
np.savez_compressed(file=f"npz/splited_training_dataset_cut-sec-{CUT_SEC}_n-mfcc-{N_MFCC}_sr-{SR}_2500.npz",
                    features=features, norm_features=norm_features, labels=labels)

# for test data
features, norm_features = dm.test_data_preprocessing(cut_sec=CUT_SEC, n_mfcc=N_MFCC, sr=SR)
print(np.shape(features), np.shape(norm_features))
np.savez_compressed(file=f"npz/test_dataset_cut-sec-{CUT_SEC}_n-mfcc-{N_MFCC}_sr-{SR}.npz",
                    features=features, norm_features=norm_features)
