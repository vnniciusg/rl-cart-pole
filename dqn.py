import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeepQNetwork(nn.Module):
    def __init__(self, lr: float, input_dims: tuple[int], fc1_dims: int, fc2_dims: int, n_actions: int) -> None:
        super(DeepQNetwork, self).__init__()
        
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        # network layers
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        return self.fc3(x)
