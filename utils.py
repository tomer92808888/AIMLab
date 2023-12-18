import os
import csv
import scipy.io as sio
from collections import Counter
from sklearn.model_selection import train_test_split
import random
import constants


def load_references(folder):
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder '{folder}' does not exist.")

    ecg_leads, ecg_labels, ecg_names = [], [], []

    labels_file = os.path.join(folder, 'labels.csv')
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Labels file '{labels_file}' not found in the folder.")

    with open(labels_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            file_name = os.path.join(folder, row[0] + '.mat')
            if not os.path.exists(file_name):
                print(f"Warning: File '{file_name}' not found.")
                continue
            data = sio.loadmat(file_name)
            ecg_leads.append(data['val'][0])
            ecg_labels.append(row[1])
            ecg_names.append(row[0])

    print(f"{len(ecg_leads)} files were loaded.")

    return ecg_leads, ecg_labels, ecg_names  # Removed FS if it's not defined elsewhere

def oversample(data, labels):
    counter = Counter(labels)
    max_count = max(counter.values())

    new_data, new_labels = [], []
    for cls in counter:
        cls_data = [datum for datum, label in zip(data, labels) if label == cls]
        while counter[cls] < max_count:
            cls_data.append(random.choice(cls_data))
            counter[cls] += 1
        new_data.extend(cls_data)
        new_labels.extend([cls] * counter[cls])

    return new_data, new_labels

def load_and_preprocess_data(file_path):
    ecg_leads, ecg_labels, ecg_names = load_references(file_path)
    fs = constants.FS
    ecg_leads, ecg_labels = oversample(ecg_leads, ecg_labels)
    return ecg_leads, ecg_labels, ecg_names, fs

def split_data(ecg_leads):
    return train_test_split(list(range(len(ecg_leads))), test_size=0.3, random_state=42, shuffle=True)
