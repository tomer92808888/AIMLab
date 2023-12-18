import torch
import torch.nn as nn

class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    def __repr__(self):
        return "Accuracy"

    def forward(self, prediction, label):
        prediction_argmax = prediction.argmax(dim=-1)
        label_argmax = label.argmax(dim=-1)
        accuracy = (prediction_argmax == label_argmax).float().mean()
        return accuracy


class F1(nn.Module):
    def __init__(self, classes=None):
        super(F1, self).__init__()
        self.classes = classes

    def __repr__(self):
        return "F1"

    def forward(self, prediction, label):
        prediction = (prediction == prediction.max(dim=-1, keepdim=True)[0]).float()
        label = (label == label.max(dim=-1, keepdim=True)[0]).float()

        class_f1_scores = []
        classes = self.classes or range(prediction.shape[-1])

        for c in classes:
            true_positive = (label[..., c] * prediction[..., c]).sum(dim=-1)
            false_positive = ((1 - label[..., c]) * prediction[..., c]).sum(dim=-1)
            false_negative = (label[..., c] * (1 - prediction[..., c])).sum(dim=-1)

            precision = true_positive / (true_positive + false_positive + 1e-8)
            recall = true_positive / (true_positive + false_negative + 1e-8)

            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            class_f1_scores.append(f1)

        return torch.stack(class_f1_scores).mean()