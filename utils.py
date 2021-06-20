from pydub.utils import mediainfo
import numpy as np
import librosa
import os


class DataModule:
    def __init__(self, data_dir_path: str, training_dir_name: str, test_dir_name: str):
        """
        :param data_dir_path: The path must end with a '/'
        :param training_dir_name: ex) "train"
        :param test_dir_name: ex) "test"
        """
        self.DATA_DIR_PATH = data_dir_path
        self.TRAIN_DATA_DIR_PATH = data_dir_path + training_dir_name + "/"
        self.TEST_DATA_DIR_PATH = data_dir_path + test_dir_name + "/"

    def training_data_preprocessing(self, cut_sec: int, n_mfcc: int, sr=48000):
        label_list = os.listdir(self.TRAIN_DATA_DIR_PATH)

        features = list()
        norm_features = list()
        labels = list()
        for label_index, label in enumerate(label_list):
            flist = os.listdir(self.TRAIN_DATA_DIR_PATH + label)
            per = len(flist) / 100
            for file_index, fname in enumerate(flist):
                audio_path = self.TRAIN_DATA_DIR_PATH + label + "/" + fname
                audio, sr = librosa.load(path=audio_path, sr=sr)

                if np.shape(audio)[0] < sr * cut_sec:
                    expand_arr = np.zeros(shape=((sr * cut_sec) - np.shape(audio)[0]), dtype="float32")
                    audio = np.append(audio, expand_arr)
                elif np.shape(audio)[0] > sr * cut_sec:
                    cutted_arr = np.split(ary=audio, indices_or_sections=(sr * cut_sec,))
                    audio = cutted_arr[0]

                audio_mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=n_mfcc)
                features.append(audio_mfcc)
                audio_mfcc_norm = librosa.util.normalize(audio_mfcc)
                norm_features.append(audio_mfcc_norm)
                labels.append(label_index)

                if (file_index % int(per)) == 0:
                    print(f"{label_index + 1}번 째 디렉터리의 파일 {file_index}개 완료, {(file_index / per)}%")

        features = np.array(features)
        norm_features = np.array(norm_features)
        labels = np.array(labels)
        print(np.shape(features), np.shape(norm_features), np.shape(labels))

        return features, norm_features, labels
