import pandas as pd
import numpy as np

result_file_paths = ["npz/test_dataset_cut-sec-10_n-mfcc-40_sr-48000_result.npz"]

for result_file_index, result_file_path in enumerate(result_file_paths):
    nploader = np.load(result_file_path)
    result = nploader["result"]
    csv_file_path = f"D:/AI/data/english accent classification/submission_{result_file_index}.csv"
    print(f"Write csv to {csv_file_path}")
    id_array = np.arange(1, 6101)
    d = {"id": id_array,
         "africa": result[:, 0],
         "australia": result[:, 1],
         "canada": result[:, 2],
         "england": result[:, 3],
         "hongkong": result[:, 4],
         "us": result[:, 5]}
    df = pd.DataFrame(d)
    df.to_csv(csv_file_path, index=False)
