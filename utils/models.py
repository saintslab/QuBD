import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    """A four-layered Multi-Layer Perceptron with width scaling."""
    def __init__(self, input_dim=784, hidden_dims=[1024, 512, 256], num_classes=10, scale=1.0):
        super(SimpleMLP, self).__init__()
        # Apply the scaling factor to the number of neurons in each hidden layer
        scaled_hidden_dims = [int(h * scale) for h in hidden_dims]
        self.layers = nn.ModuleList()
        curr_dim = input_dim
        for h_dim in scaled_hidden_dims:
            self.layers.append(nn.Linear(curr_dim, h_dim))
            curr_dim = h_dim
        self.final_layer = nn.Linear(curr_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            # Linear transformation followed by ReLU activation
            x = F.relu(layer(x))
        x = self.final_layer(x)
        return x

