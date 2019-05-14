import torch
import numpy as np
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """Network constructor"""
        super(Actor, self).__init__()
        #Initialize model here

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.nl1 = nn.Tanh()

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.nl2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.nl3 = nn.LeakyReLU()



    def forward(self, x):
        """Network forward pass"""
        if type(x) is np.ndarray:
            x = torch.FloatTensor(x)


        y = self.fc1(x)
        y = self.nl1(y)

        y = self.fc2(y)
        y = self.nl2(y)

        y = self.fc3(y)
        y = self.nl3(y)

        return y.view(-1,self.input_dim, self.input_dim)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden, output_dim):
        """Network constructor"""
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = output_dim
        self.cat_dim = hidden + action_dim


        self.fc1 = nn.Linear(state_dim, hidden)
        self.nl1 = nn.Tanh()

        self.fc2 = nn.Linear(self.cat_dim, hidden)
        self.nl2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden, output_dim)

    def forward(self, state, action):

        state_half = self.fc1(state)
        state_half = self.nl2(state_half)
        action_half = action.view(-1,self.action_dim)


        combined = torch.cat([state_half,action_half], dim=1)

        y = self.fc2(combined)
        y = self.nl2(y)

        return self.fc3(y)



