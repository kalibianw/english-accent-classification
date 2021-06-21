from utils import TrainModule
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import shutil
import os


NPZ_PATH = "npz/training_dataset_cut-sec-10_n-mfcc-20_sr-48000.npz"
NPZ_NAME = os.path.basename(NPZ_PATH)

BATCH_SIZE = 32
EPOCHS = 1000
CKPT_PATH = f"D:/AI/ckpt/english-accent-classification/{NPZ_NAME}/"
if os.path.exists(CKPT_PATH):
    shutil.rmtree(CKPT_PATH)
    os.mkdir(CKPT_PATH)
elif os.path.exists(CKPT_PATH) is False:
    os.mkdir(CKPT_PATH)
MODEL_PATH = f"D:/AI/model/english-accent-classification/{NPZ_NAME}.h5"
LOG_DIR = f"logs/{NPZ_NAME}/"
if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)
    os.mkdir(LOG_DIR)
elif os.path.exists(LOG_DIR) is False:
    os.mkdir(LOG_DIR)

nploader = np.load(NPZ_PATH)
norm_features, labels = nploader["norm_features"], nploader["labels"]
print(np.shape(norm_features), np.shape(labels), np.unique(labels, return_counts=True))

norm_features = np.expand_dims(norm_features, axis=-1)
labels = to_categorical(labels)
print(np.shape(norm_features), np.shape(labels))

x_train, x_valid, y_train, y_valid = train_test_split(norm_features, labels, test_size=0.2)

tm = TrainModule(batch_size=BATCH_SIZE,
                 ckpt_path=CKPT_PATH, model_path=MODEL_PATH,
                 input_shape=np.shape(norm_features[0, :, :, :]),
                 log_dir=LOG_DIR)

model = tm.create_model()
tm.training_model(model=model,
                  x_train=x_train, y_train=y_train,
                  x_valid=x_valid, y_valid=y_valid,
                  epochs=EPOCHS)
