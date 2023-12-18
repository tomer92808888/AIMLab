from pade_activation_unit import PADEACTIVATION_Function_based

FS = 300

PAU = PADEACTIVATION_Function_based

PREPROCESSING_PIPELINE_CONFIG = {
    "p_scale": 0.2,
    "p_drop": 0.2,
    "p_cutout": 0.2,
    "p_shift": 0.2,
    "p_resample": 0.2,
    "p_random_resample": 0.2,
    "p_sine": 0.2,
    "p_band_pass_filter": 0.2,
    "scale_range": (0.85, 1.15),
    "drop_rate": 0.03,
    "interval_length": 0.05,
    "max_shift": 4000,
    "resample_factors": (0.8, 1.2),
    "max_offset": 0.075,
    "resampling_points": 12,
    "max_sine_magnitude": 0.3,
    "sine_frequency_range": (.2, 1.),
    "kernel": (1, 6, 15, 20, 15, 6, 1),
    "ecg_sequence_length": 18000,
    "fs": 300,
    "frequencies": (0.2, 45.)
}

MODEL_CONFIG = {
    "ecg_features": 256,
    "transformer_heads": 8,
    "transformer_ff_features": 512,
    "transformer_activation": "gelu",
    "transformer_layers": 3,
    "transformer_sequence_length": 80,
    "spectrogram_encoder_channels": ((1, 128), (128, 256), (256, 512), (512, 512), (512, 256)),
    "spectrogram_encoder_spans": (None, None, (140, 8), (70, 4), (35, 2)),
    "latent_vector_features": 256,
    "classes": 4,
    "dropout": 0.05,
    "activation": PAU,
}