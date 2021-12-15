import numpy as np
from tensorflow.keras import models
import os

N_MFCCS = [40]
BATCH_SIZE = 32
MODEL_PATHS = [
    "D:/AI/model/english-accent-classification/training_dataset_cut-sec-10_n-mfcc-40_sr-48000.h5",
]

for index, n_mfcc in enumerate(N_MFCCS):
    npz_file_path = f"npz/test_dataset_cut-sec-10_n-mfcc-{n_mfcc}_sr-48000.npz"
    fname = os.path.splitext(os.path.basename(npz_file_path))[0]
    nploader = np.load(npz_file_path)
    norm_features = np.expand_dims(nploader["norm_features"], axis=-1)
    model = models.load_model(MODEL_PATHS[index])
    result = model.predict(norm_features, batch_size=BATCH_SIZE, verbose=1)
    np.savez_compressed("npz/" + fname + "_result.npz", result=result)
