import torch
from tqdm import tqdm
from Model import Model
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np

class Test:
    def __init__(self, model_path, model_config, test_dataset):
        self.model = self._initialize_model(model_path, model_config)
        self.test_dataset = test_dataset

    def _initialize_model(self, model_path, model_config):
        model = Model(config=model_config)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        return model

    def test_model(self, test_dataset, ecg_names, generate_csv=False, plot_graphs=False):
        self.model.eval()
        self.model.to(torch.device('cpu'))

        predictions = []
        progress_bar = tqdm(total=len(test_dataset))
        for name, data in zip(ecg_names, test_dataset):
            progress_bar.update(n=1)

            ecg_lead, spectrogram, label = data
            ecg_lead, spectrogram = ecg_lead.to('cpu'), spectrogram.to('cpu')

            prediction = self.model(ecg_lead, spectrogram).argmax(dim=-1)
            label_argmax = label.argmax(dim=-1)

            predictions.append((name, self._get_prediction_label(prediction), self._get_prediction_label(label_argmax)))

        progress_bar.close()

        if generate_csv:
            self._generate_csv(predictions)

        y_true = [x[2] for x in predictions]
        y_pred = [x[1] for x in predictions]
        print(classification_report(y_true, y_pred, labels=["N", "O", "A", "~"]))

        if plot_graphs:
            self._plot_confusion_matrix(y_true, y_pred)

        return predictions

    def _generate_csv(self, predictions):
        df = pd.DataFrame(predictions, columns=["File", "Prediction", "True Label"])
        df.to_csv("test_predictions.csv", index=False)

    def _plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=["N", "O", "A", "~"])
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_rounded = np.round(cm_normalized, 2)
        df_cm = pd.DataFrame(cm_rounded, index=["N", "O", "A", "~"], columns=["N", "O", "A", "~"])
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, cmap="Blues")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()

    def _get_prediction_label(self, prediction):
        prediction = int(prediction.item())
        label_map = {0: "N", 1: "O", 2: "A", 3: "~"}
        return label_map.get(prediction, "Invalid Prediction")