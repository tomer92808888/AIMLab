import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram

class PhysioNetDataset(Dataset):
    def __init__(self, ecg_leads, ecg_labels,
                 augmentation_pipeline=None, spectrogram_length=563,
                 ecg_sequence_length=18000, ecg_window_size=256, ecg_step=224,
                 normalize=True, fs=300, spectrogram_n_fft=64, spectrogram_win_length=64,
                 spectrogram_power=1, spectrogram_normalized=True):
        super(PhysioNetDataset, self).__init__()

        self.ecg_leads = [torch.from_numpy(data).float() for data in ecg_leads]
        self.augmentation_pipeline = augmentation_pipeline or nn.Identity()
        self.spectrogram_length = spectrogram_length
        self.ecg_sequence_length = ecg_sequence_length
        self.ecg_window_size = ecg_window_size
        self.ecg_step = ecg_step
        self.normalize = normalize
        self.fs = fs
        self.classes = 4

        self.ecg_labels = self.__process_labels(ecg_labels)

        self.spectrogram_module = Spectrogram(n_fft=spectrogram_n_fft, win_length=spectrogram_win_length,
                                              hop_length=spectrogram_win_length // 2, power=spectrogram_power,
                                              normalized=spectrogram_normalized)

    def __len__(self):
        return len(self.ecg_leads)

    def __getitem__(self, item):
        ecg_lead, ecg_label = self.ecg_leads[item], self.ecg_labels[item]
        ecg_lead = self.__preprocess_ecg_lead(ecg_lead)
        spectrogram = self.__compute_spectrogram(ecg_lead)
        ecg_lead = self.__unfold_ecg_lead(ecg_lead)
        ecg_label = self.__one_hot_encode(ecg_label)

        return ecg_lead.float(), spectrogram.unsqueeze(dim=0).float(), ecg_label

    def __process_labels(self, ecg_labels):
        label_mapping = {"N": 0, "O": 1, "A": 2, "~": 3}
        return [torch.tensor(label_mapping[label], dtype=torch.long) for label in ecg_labels]

    def __preprocess_ecg_lead(self, ecg_lead):
        ecg_lead = self.augmentation_pipeline(ecg_lead[:self.ecg_sequence_length])
        if self.normalize:
            ecg_lead = (ecg_lead - ecg_lead.mean()) / (ecg_lead.std() + 1e-08)
        return F.pad(ecg_lead, pad=(0, self.ecg_sequence_length - ecg_lead.shape[0]), value=0., mode="constant")

    def __compute_spectrogram(self, ecg_lead):
        spectrogram = self.spectrogram_module(ecg_lead)
        spectrogram = torch.log(spectrogram.abs().clamp(min=1e-08))
        return F.pad(spectrogram, pad=(0, self.spectrogram_length - spectrogram.shape[-1]),
                     value=0., mode="constant").permute(1, 0)

    def __unfold_ecg_lead(self, ecg_lead):
        return ecg_lead.unfold(dimension=-1, size=self.ecg_window_size, step=self.ecg_step)

    def __one_hot_encode(self, ecg_label):
        return F.one_hot(ecg_label, num_classes=self.classes)
