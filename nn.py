import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        """Network constructor"""
        super(Network, self).__init__()
        #Initialize model here
        self.model = None


        self.loss_fn = None
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)


    def forward(self, input):
        """Network forward pass"""
        return None


    def train(self,state, requests, optimal):
        """
        Main training loop for the learning agent that learns the pricing vector function

        state : torch
            the current state, which is some vector of length n integers where each ith element of the vector
            represents the number of drivers at that location i.
        requests: torch
            the incoming requests, should be some nxn by grid of integers with the number of requests going from
            location i to j

        optimal: torch
            some nxn grid of real numbers indicating the optimal pricing for a trip from location
            i to location j

        Doesn't return anything, just trains the function approximator
        """

        pricing = self.forward(state)

        self.optimizer.zero_grad()  # zero the gradient buffers
        loss = self.loss_fn(pricing, optimal)
        loss.backwards() #backwards pass
        self.optimizer.step() #update network



