import torch
import torch.nn as nn


# Define the model
class SoftmaxRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SoftmaxRegression, self).__init__()
        self.linear = nn.Linear(int(torch.prod(torch.tensor(input_size))), num_classes)

    def forward(self, x):
        logits = self.linear(x)
        return nn.functional.softmax(logits, dim=1)
