import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicMLP(nn.Module):
    def __init__(
        self, 
        input_size: int = 28*28, 
        hidden_size: int = 400, 
        output_size: int = 10, 
        vdo_rate: float = 0.20,
        hdo_rate: float = 0.50
    ) -> None:
        super(BasicMLP, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.do1 = nn.Dropout(vdo_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.do2 = nn.Dropout(hdo_rate)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.do3 = nn.Dropout(hdo_rate)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self._activation_hooks = {}
    
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc1(x))
        x = self.do1(x)
        x = F.relu(self.fc2(x))
        self._activation_hooks['h1'] = x.detach().cpu().numpy()
        x = self.do2(x)
        x = F.relu(self.fc3(x))
        self._activation_hooks['h2'] = x.detach().cpu().numpy()
        x = self.do3(x)
        x = self.fc4(x)
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

def mnist800(**kwargs) -> BasicMLP:
    """Wrapper function for constructing MNIST student model."""
    return _mlp(28*28, 800, 10, **kwargs)

if __name__ == '__main__':
    model = mnist1200() 
    sample = torch.randn(1, 1, 28, 28)
    output = model(sample)
    print(output.shape) 