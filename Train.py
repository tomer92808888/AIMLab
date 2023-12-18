import torch
import torch_optimizer
from tqdm import tqdm
import wandb
from SoftmaxCrossEntropyLoss import SoftmaxCrossEntropyLoss
from Model import Model
from metric import Accuracy, F1

class Train:
    def __init__(self, weights_file, model_config, device):
        self.__device = device
        self.model = self._initialize_model(weights_file, model_config)

    def _initialize_model(self, weights_file, model_config):
        model = Model(config=model_config)
        state_dict = torch.load(weights_file, map_location=torch.device(self.__device))

        # Filter out incompatible keys
        model_state_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
        model_state_dict.update(state_dict)
        model.load_state_dict(model_state_dict)

        self.optimizer = torch_optimizer.RAdam(params=model.parameters(), lr=1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self.optimizer, milestones=[25, 50, 75], gamma=0.1)
        self.loss_function = SoftmaxCrossEntropyLoss((0.4, 0.7, 0.9, 0.9))
        return model

    def train_model(self, train_dataset, val_dataset, epochs, validate_every_n_epochs):
        self.model.train()
        self.model.to(self.__device)
        best_validation_metric = 0.0
        progress_bar = tqdm(total=epochs * len(train_dataset))
        run = wandb.init(project="Technion-PhysioNet")

        for epoch in range(epochs):
            run.log({"epoch": epoch})

            for ecg_leads, spectrogram, labels in train_dataset:
                progress_bar.update(n=1)
                self.optimizer.zero_grad()

                ecg_leads, spectrogram, labels = ecg_leads.to(self.__device), spectrogram.to(self.__device), labels.to(self.__device)
                predictions = self.model(ecg_leads, spectrogram)
                loss = self.loss_function(predictions, labels)
                loss.backward()
                self.optimizer.step()

                progress_bar.set_description(f"Epoch {epoch + 1}/{epochs} Loss={loss.item():.4f} Best Val Metric={best_validation_metric:.4f}")
                run.log({"train_loss": loss.item()})

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if (epoch + 1) % validate_every_n_epochs == 0 or epoch == epochs - 1:
                current_validation_metric = self.validate_model(val_dataset)
                if current_validation_metric > best_validation_metric:
                    best_validation_metric = current_validation_metric
                    torch.save(self.model.state_dict(), "best_finetune_model.pt")

            run.log({"best_validation_metric": best_validation_metric})

        progress_bar.close()

    @torch.no_grad()
    def validate_model(self, val_dataset, validation_metrics=(F1(), Accuracy())):
        self.model.eval()
        self.model.to(self.__device)

        predictions, labels = [], []
        for ecg_leads, spectrogram, batch_labels in val_dataset:
            ecg_leads, spectrogram, batch_labels = ecg_leads.to(self.__device), spectrogram.to(self.__device), batch_labels.to(self.__device)
            batch_predictions = self.model(ecg_leads, spectrogram)
            predictions.append(batch_predictions)
            labels.append(batch_labels)

        predictions = torch.cat(predictions, dim=0)
        labels = torch.cat(labels, dim=0)

        metric_values = {}
        for metric in validation_metrics:
            metric_value = metric(predictions, labels)
            metric_values[f"val_{type(metric).__name__}".lower()] = metric_value

        self.model.train()
        wandb.log(metric_values)
        return metric_values["val_accuracy"]
