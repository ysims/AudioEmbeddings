import os
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.dataset_utils import wavfile_to_examples


class ZSLFSC22Dataset(Dataset):
    # data_path: to a pickle file with
    #   data["audio"]
    #   data["labels"]
    def __init__(self, train_classes, device="cuda", channels=1, bins=64, data_path="../data/FSC22/audio"):
        self.audio = []
        self.labels = []
        self.device = device

        for file in os.listdir(data_path):
            # Get the class from the file name
            target = int(file.split("_")[0])
            
            # Check if we want to train with this class
            if train_classes.count(target) > 0:
                wav_data = wavfile_to_examples(os.path.join(data_path, file), bins)
                if channels == 3:
                    self.audio.append(np.array([wav_data[0], wav_data[0], wav_data[0]]))
                elif channels == 1:
                    self.audio.append(wav_data)
                self.labels.append(
                    train_classes.index(target)
                )  # only have numbers in the range of [0, len(train_classes)]
        self.length = len(self.audio)

        self.audio = np.array(self.audio)
        self.labels = np.array(self.labels)

        self.audio = torch.from_numpy(self.audio).float().to(self.device)
        self.labels = torch.from_numpy(self.labels).long().to(self.device)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return (self.audio[index], self.labels[index])
