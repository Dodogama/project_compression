import torch.nn as nn

class BasicMLP(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=400, output_size=10):
        super(BasicMLP, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.relu1(self.fc1(x))
        x = self.do1(x)
        x = self.relu2(self.fc2(x))
        x = self.do2(x)
        x = self.fc3(x)
        return x