import torch
import torch.nn as nn

class SoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, weight=(1., 1., 1., 1.)):
        super(SoftmaxCrossEntropyLoss, self).__init__()

        self.register_buffer('weight', torch.tensor(weight, dtype=torch.float32).view(1, -1))

    def forward(self, prediction, label):
        self.weight = self.weight.to(prediction.device)
        loss = - (self.weight * label * torch.log_softmax(prediction, dim=-1)).mean()
        return loss