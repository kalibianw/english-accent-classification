import numpy as np
from tensorflow.keras import models

N_MFCCS = [40, 80]
BATCH_SIZE = 32
MODEL_PATHS = [
    "D:/AI/model/english-accent-classification/training_dataset_cut-sec-10_n-mfcc-40_sr-48000.h5",
    "D:/AI/model/english-accent-classification/splited_training_dataset_cut-sec-10_n-mfcc-80_sr-48000_2500.h5"
]

for index, n_mfcc in enumerate(N_MFCCS):
    npz_file_path = f"test_dataset_cut-sec-10_n-mfcc-{n_mfcc}_sr-48000.npz"
    nploader = np.load(npz_file_path)
    norm_features = np.expand_dims(nploader["norm_features"], axis=-1)
    model = models.load_model(MODEL_PATHS[index])
    result = model.predict(norm_features, batch_size=BATCH_SIZE, verbose=1)
    print(result)
    print(np.shape(result))
