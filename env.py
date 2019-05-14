import memory
import numpy as np
import torch
from torch.autograd import Variable

import lp

BUFFER_SIZE = 100
MAX_CARS = 3
MAX_REQUEST = 3


class Env():

    def __init__(self, locations, actor):
        self.locations = locations
        self.grid_size = (locations, locations)

        self.buffer = memory.Replay(buffer_size=BUFFER_SIZE)

        self.curr_state = MAX_CARS*np.ones(self.locations)
        self.transit_times = np.random.randint(0,3,size=self.grid_size)
        self.max_prices = self.max_pricing()

        self.actor = actor
        self.actor.eval()


    def populate(self):
        self.actor.eval()
        for i in range(BUFFER_SIZE):

            # state = np.random.randint(0,MAX_CARS,self.locations)
            state = MAX_CARS*np.ones(self.locations)

            state_tensor = torch.Tensor(state)
            pricing_tensor = self.actor(state_tensor)

            pricing = pricing_tensor.data.numpy().reshape(self.grid_size)
            requests = np.random.randint(0, MAX_REQUEST, size=self.grid_size)
            matched, leftover = lp.match(env=self, state=state, pricing=pricing, request=requests)
            # reward = np.sum(np.multiply(pricing, matched))
            reward = self.compute_reward(pricing,matched,leftover)
            depart_sum = np.sum(matched, axis=1)
            arrive_sum = np.sum(matched, axis=0)
            next_state = state - depart_sum + arrive_sum

            self.buffer.add(s=state, a=pricing, s2=next_state, r=reward)

    def add_memory(self, state, action, reward, next_state):
            #self.buffer.add(s=state[i], a=action[i], r=reward[i], s2=next_state[i])
        if type(action) is torch.Tensor:
            action = action.data.numpy().reshape(self.grid_size)

        self.buffer.add(s=state, a=action, r=reward, s2=next_state)

    def observe(self,batch_size=10):
        states, actions, rewards, next_states = self.buffer.sample_batch(batch_size=batch_size)

        states = torch.Tensor(states)
        next_states = torch.Tensor(next_states)
        actions = torch.Tensor(actions)
        rewards = torch.Tensor(rewards)

        return states, actions, rewards, next_states

    def state(self):
        return self.curr_state

    def state_tensor(self):
        tensor = torch.Tensor(self.curr_state)
        tensor = tensor.view(-1, self.locations)
        return tensor

    def max_tenor(self):
        tensor = torch.Tensor(self.max_pricing)
        size = (-1,) + self.grid_size
        tensor = tensor.view(size)
        return tensor

    def compute_reward(self, pricing, matched, leftover):
        pricing = np.maximum(pricing,0)
        gain = np.sum(np.multiply(matched,pricing))
        loss = np.sum(np.multiply(leftover, pricing))
        return gain #- loss

    def step(self, pricing):
        pricing = pricing.data.numpy().reshape(self.grid_size)
        requests = np.random.randint(0, MAX_REQUEST, size=self.grid_size)
        matched, leftover = lp.match(env=self, pricing=pricing, request=requests)
        #reward = np.sum(np.multiply(pricing, matched))
        reward = self.compute_reward(pricing,matched,leftover)

        depart_sum = np.sum(matched, axis=1)
        arrive_sum = np.sum(matched, axis=0)
        next_state = self.curr_state - depart_sum + arrive_sum
        self.curr_state = next_state
        return next_state, reward

    def wtp(self, pricing):
        diff = self.max_prices-pricing
        return 1 - np.abs(np.tanh(diff))

    def wtp_loss(self, pricing):
        max_pricing = Variable(torch.Tensor(self.max_prices), requires_grad=False)
        diff = torch.abs(max_pricing-pricing)
        return torch.sum(1 - torch.tanh(diff))

    def max_pricing(self):
        ix = np.indices(self.grid_size)
        dist = np.power(ix[0]-ix[1],2)
        norm = np.sqrt(dist)
        return norm

    def update(self, next):
        self.curr_state = next



