import sys
import nn
import numpy as np
import env as environment
from torch.optim import Adam
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gym_dynamic_set_packing


MAX_EPISODES = 30
STEPS = 100
BATCH_SIZE = 1


NUM_TYPES = 16
ACTOR_HIDDEN = 128
CRTIC_HIDDEN = 128
CRITIC_OUTPUT = 1
EXPLORATION = .5
ACTOR_OUTPUT = NUM_TYPES
DISCOUNT = .999
TAU = 1e-3
CRITIC_LR = 1e-5
ACTOR_LR = 1e-4

class GreedyActor():
    def eval(self):
        return
    def __call__(self, state):
        return torch.ones(1,NUM_TYPES)

    def train(self):
        return

def greedy(args):
    #Randomly initialize actor and critic networks
    #Copy the weights to make the targets

    actor = GreedyActor()
    env = environment.EnvKidney('DynamicSetPacking-weightedtest-v0', actor= actor)
    env.populate() #Initialize buffer


    prices_l = []
    critic_l = []
    actual_rewards = []
    for e in range(MAX_EPISODES):
        env.reset()
        for i in range(STEPS):

            state = env.state()
            pricing = actor(state)

            next_state, reward = env.step(pricing)

            env.add_memory(state=state, action=pricing, reward=reward, next_state=next_state)

            states, actions, rewards, nexts = env.observe(batch_size=100)
            actual_rewards.append(reward)



        env.update(next_state)
    return actual_rewards


def main(args):

    #Randomly initialize actor and critic networks
    actor = nn.Actor(input_dim=NUM_TYPES, hidden_dim=ACTOR_HIDDEN, output_dim=ACTOR_OUTPUT)
    actor_optim = Adam(actor.parameters(), lr=ACTOR_LR)

    critic = nn.Critic(state_dim=NUM_TYPES,action_dim=ACTOR_OUTPUT,hidden=CRTIC_HIDDEN,output_dim=CRITIC_OUTPUT)
    critic_optim = Adam(critic.parameters(), lr=CRITIC_LR)

    #Copy the weights to make the targets
    actor_target = nn.Actor(input_dim=NUM_TYPES, hidden_dim=ACTOR_HIDDEN, output_dim=ACTOR_OUTPUT)
    critic_target = nn.Critic(state_dim=NUM_TYPES,action_dim=ACTOR_OUTPUT,hidden=CRTIC_HIDDEN,output_dim=CRITIC_OUTPUT)

    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())


    env = environment.EnvKidney('DynamicSetPacking-weightedtest-v0', actor= actor)
    env.populate() #Initialize buffer


    prices_l = []
    critic_l = []
    actual_rewards = []

    for e in range(MAX_EPISODES):
        env.reset()
        for i in range(STEPS):
            if actor.training != True:
                actor.train()
            if critic.training != True:
                critic.train()

            actor_target.eval()
            critic_target.eval()

            state = env.state()
            pricing = actor(state)
            pricing_noised = pricing + EXPLORATION*torch.randn(size=pricing.size())

            next_state, reward = env.step(pricing_noised)

            env.add_memory(state=state, action=pricing_noised, reward=reward, next_state=next_state)

            states, actions, rewards, nexts = env.observe(batch_size=100)
            pricing_target = actor_target(states) #+ 10*torch.rand(size=pricing.size())
            actual_rewards.append(reward)


            critic_optim.zero_grad()

            rewards_tenosr = torch.Tensor(rewards).unsqueeze(dim=1)
            pred_future_reward = DISCOUNT*critic_target(nexts,pricing_target)

            y_i = rewards_tenosr + pred_future_reward
            y = critic(states, actions)

            critic_loss = F.l1_loss(y_i,y)
            critic_l.append(reward)

            critic_loss.backward()
            critic_optim.step()

            actor_optim.zero_grad()

            pricing = actor(states)

            policy_loss = -critic(states, pricing) #- env.wtp_loss(pricing)
            policy_loss = policy_loss.mean()
            policy_loss.backward()
            actor_optim.step()
            critic.eval()

            soft_update(actor_target, actor, TAU)
            soft_update(critic_target, critic, TAU)


        env.update(next_state)

    return actual_rewards


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def running_mean(x, N):
    # from stackoverflow
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

if __name__ == '__main__':
    rewards_learned = main(sys.argv)
    rewards_greedy = greedy(sys.argv)
    plt.plot(running_mean(rewards_learned, 10))
    plt.plot(running_mean(rewards_greedy, 10))
    plt.show()
