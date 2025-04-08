import torch
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

def _mlp(input_size: int, hidden_size: int, output_size: int, **kwargs) -> BasicMLP:
    """Constructs a BasicMLP model."""
    model = BasicMLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    pretrained = kwargs.get('pretrained', None)
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model

def mnist1200(**kwargs) -> BasicMLP:
    """Wrapper function for constructing MNIST teacher model."""
    return _mlp(28*28, 1200, 10, **kwargs)


def mnist400(**kwargs) -> BasicMLP:
    """Wrapper function for constructing MNIST student model."""
    return _mlp(28*28, 400, 10, **kwargs)

if __name__ == '__main__':
    model = mnist1200() 
    sample = torch.randn(1, 1, 28, 28)
    output = model(sample)
    print(output.shape) 