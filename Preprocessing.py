import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal

class Preprocessing(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, ecg_lead):
        for augment, prob in self.config.items():
            if augment.startswith("p_") and random.random() <= prob:
                ecg_lead = getattr(self, augment[2:])(ecg_lead)
        return ecg_lead

    def scale(self, ecg_lead):
        scale_range = self.config['scale_range']
        random_scalar = torch.rand(1).float() * (scale_range[1] - scale_range[0]) + scale_range[0]
        return random_scalar * ecg_lead

    def drop(self, ecg_lead):
        num_dropped_samples = int(ecg_lead.shape[-1] * self.config['drop_rate'])
        dropped_indices = torch.randperm(ecg_lead.shape[-1])[:max(1, num_dropped_samples)]
        ecg_lead = ecg_lead.clone()  # To avoid modifying the original tensor
        ecg_lead[..., dropped_indices] = 0.
        return ecg_lead

    def cutout(self, ecg_lead):
        interval_size = int(ecg_lead.shape[-1] * self.config['interval_length'])
        index_start = torch.randint(0, ecg_lead.shape[-1] - interval_size, (1,))
        ecg_lead = ecg_lead.clone() # To avoid modifying the original tensor
        ecg_lead[index_start:index_start + interval_size] = 0.
        return ecg_lead

    def shift(self, ecg_lead):
        shift = torch.randint(0, self.config['max_shift'], (1,))
        shifted_lead = torch.cat([torch.zeros_like(ecg_lead)[..., :shift], ecg_lead], dim=-1)
        return shifted_lead[..., :self.config['ecg_sequence_length']]

    def resample(self, ecg_lead):
        resample_factor = torch.rand(1).float() * (self.config['resample_factors'][1] - self.config['resample_factors'][0]) + self.config['resample_factors'][0]
        resampled_lead = F.interpolate(ecg_lead[None, None], size=int(resample_factor * ecg_lead.shape[-1]), mode="linear", align_corners=False)[0, 0]
        return resampled_lead[..., :self.config['ecg_sequence_length']]

    def random_resample(self, ecg_lead):
        coordinates = 2. * (torch.arange(ecg_lead.shape[-1]).float() / (ecg_lead.shape[-1] - 1)) - 1
        offsets = F.interpolate(((2 * torch.rand(self.config['resampling_points']) - 1) * self.config['max_offset'])[None, None], size=ecg_lead.shape[-1], mode="linear", align_corners=False)[0, 0]
        grid = torch.stack([coordinates + offsets, coordinates], dim=-1)[None, None].clamp(min=-1, max=1)
        resampled_lead = F.grid_sample(ecg_lead[None, None, None], grid=grid, mode='bilinear', align_corners=False)[0, 0, 0]
        return resampled_lead[..., :self.config['ecg_sequence_length']]

    def sine(self, ecg_lead):
        sine_magnitude = torch.rand(1).float() * self.config['max_sine_magnitude']
        sine_frequency = torch.rand(1).float() * (self.config['sine_frequency_range'][1] - self.config['sine_frequency_range'][0]) + self.config['sine_frequency_range'][0]
        t = torch.arange(ecg_lead.shape[-1]) / float(self.config['fs'])
        sine = torch.sin(2 * math.pi * sine_frequency * t + torch.rand(1)) * sine_magnitude
        return sine + ecg_lead

    def band_pass_filter(self, ecg_lead):
        sos = signal.butter(10, self.config['frequencies'], 'bandpass', fs=self.config['fs'], output='sos')
        filtered_lead = torch.from_numpy(signal.sosfilt(sos, ecg_lead.numpy()))
        return filtered_lead