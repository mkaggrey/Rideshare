import nn
import torch
EPISODES = 100
GRID_DIM  = (100,100)
STEPS_PER_EPISODE = 1000



class Env():
    def __init__(self):
        """Initialize the environment"""
        self.env = None
        self.state = torch.empty(GRID_DIM)
        self.nn = nn.Model()

    def populate(self):
        """Initialize the state grid"""
        return

    def learn(self):
        """Learning the function approximation for price vectors"""
        #requests = generateRequestsSomehow(STEPS_PER_EPISODE)
        requests = None
        for r in requests:
            state_local = self.state
            loss = self.nn.train(state_local,r)

        return

    def step(self):
        """Change the global state and update required variables"""
        return

    def simulate(self):
        """Method resposible for simulation ridesharing envrionment: learning and training"""
        for i in range(EPISODES):
            self.learn()
            self.step()

        return



