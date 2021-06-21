# For DataModule
import numpy as np
import librosa
import os

# For TrainModule
from tensorflow.keras import models, layers, activations, initializers, optimizers, losses, callbacks


class DataModule:
    def __init__(self, data_dir_path: str, training_dir_name: str, test_dir_name: str):
        """
        :param data_dir_path: The path must end with a '/'. ex) D:/data/
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


class TrainModule:
    def __init__(self, batch_size: int, ckpt_path: str, model_path: str, input_shape: tuple, log_dir: str):
        """
        :param batch_size: ex) 32
        :param ckpt_path: ex) D:/test.ckpt
        :param model_path: ex) D:/test.h5
        :param input_shape: ex) (32, 32, 1)
        :param log_dir: The path must end with a '/'. ex) D:/test/
        """
        self.BATCH_SIZE = batch_size
        self.CKPT_PATH = ckpt_path
        self.MODEL_PATH = model_path
        self.INPUT_SHAPE = input_shape
        self.LOG_DIR = log_dir

    def create_model(self):
        model = models.Sequential([
            layers.InputLayer(input_shape=self.INPUT_SHAPE),
            layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation=activations.relu,
                          kernel_initializer=initializers.he_normal()),
            layers.MaxPooling2D(padding="same"),
            layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation=activations.swish,
                          kernel_initializer=initializers.he_normal()),
            layers.MaxPooling2D(padding="same"),
            layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation=activations.swish,
                          kernel_initializer=initializers.he_normal()),
            layers.MaxPooling2D(padding="same"),
            layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation=activations.swish,
                          kernel_initializer=initializers.he_normal()),
            layers.MaxPooling2D(padding="same"),

            layers.Flatten(),

            layers.Dense(1024, activation=activations.swish, kernel_initializer=initializers.he_normal()),
            layers.Dropout(rate=0.5),
            layers.Dense(512, activation=activations.swish, kernel_initializer=initializers.he_normal()),
            layers.Dropout(rate=0.5),
            layers.Dense(256, activation=activations.swish, kernel_initializer=initializers.he_normal()),
            layers.Dropout(rate=0.5),
            layers.Dense(128, activation=activations.swish, kernel_initializer=initializers.he_normal()),

            layers.Dense(6, activation=activations.softmax, kernel_initializer=initializers.he_normal())
        ])

        model.compile(optimizer=optimizers.Adam(), loss=losses.categorical_crossentropy, metrics=["accuracy"])

        return model

    def training_model(self,
                       model: models.Sequential,
                       x_train: np.ndarray,
                       y_train: np.ndarray,
                       x_valid: np.ndarray,
                       y_valid: np.ndarray,
                       epochs=1000):
        callback_list = [
            callbacks.EarlyStopping(min_delta=1e-4, patience=30, verbose=1),
            callbacks.ModelCheckpoint(filepath=self.CKPT_PATH,
                                      verbose=1,
                                      save_best_only=True,
                                      save_weights_only=True,),
            callbacks.ReduceLROnPlateau(factor=0.6, patience=5, verbose=1, min_lr=1e-6),
            callbacks.TensorBoard(log_dir=self.LOG_DIR)
        ]
        model.fit(
            x=x_train, y=y_train,
            batch_size=self.BATCH_SIZE,
            epochs=epochs,
            callbacks=callback_list,
            validation_data=(x_valid, y_valid)
        )

        model.load_weights(self.CKPT_PATH)
        model.save(self.MODEL_PATH)
