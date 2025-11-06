import torch
import torch.nn as nn

class NeuralNet_Binary(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet_Binary, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(x)
        out = self.sigmoid(out)
        return out
    
model = NeuralNet_Binary(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss()

class NeuralNet_MultiClass(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes) -> None:
        super(NeuralNet_MultiClass, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def froward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(x)
        return out
    
model = NeuralNet_MultiClass(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss()