from torch.utils.data import DataLoader
from Dataset import PhysioNetDataset
from constants import PREPROCESSING_PIPELINE_CONFIG
from Train import Train
from Test import Test
from constants import MODEL_CONFIG
from utils import load_and_preprocess_data, split_data
from Preprocessing import Preprocessing


def prepare_data(train, data_path="challenge_data"):
    ecg_leads, ecg_labels, ecg_names, fs = load_and_preprocess_data(data_path)
    training_split, val_split = split_data(ecg_leads)

    if train:
        training_dataset = DataLoader(
            PhysioNetDataset(ecg_leads=[ecg_leads[index] for index in training_split],
                             ecg_labels=[ecg_labels[index] for index in training_split], fs=fs,
                             augmentation_pipeline=Preprocessing(PREPROCESSING_PIPELINE_CONFIG)),
            batch_size=24, num_workers=20, pin_memory=True, drop_last=False, shuffle=True)

        validation_dataset = DataLoader(
            PhysioNetDataset(ecg_leads=[ecg_leads[index] for index in val_split],
                             ecg_labels=[ecg_labels[index] for index in val_split], fs=fs,
                             augmentation_pipeline=None),
            batch_size=24, num_workers=20, pin_memory=True, drop_last=False, shuffle=False)

        return training_dataset, validation_dataset
    else:
        validation_dataset = DataLoader(
            PhysioNetDataset(ecg_leads=[ecg_leads[index] for index in val_split],
                             ecg_labels=[ecg_labels[index] for index in val_split], fs=fs,
                             augmentation_pipeline=None),
            batch_size=1, num_workers=0, pin_memory=False, drop_last=False, shuffle=False)

        return validation_dataset, ecg_names

if __name__ == "__main__":
    train = False
    pretrained_model = "Icentia11k_pretrained.pt"
    best_model = "best_finetune_model_150.pt"

    if train:
        training_dataset, validation_dataset = prepare_data(train)
        model = Train(pretrained_model, MODEL_CONFIG, "mps")
        model.train_model(training_dataset, validation_dataset, 150, 1)
    else:
        validation_dataset, ecg_names = prepare_data(train)
        model = Test(best_model, MODEL_CONFIG, validation_dataset)
        pred = model.test_model(validation_dataset, ecg_names, generate_csv=False, plot_graphs=True)
