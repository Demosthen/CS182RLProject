import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from pponetwork import PPONetwork
import gym3
from procgen import ProcgenGym3Env
import numpy as np
import torch.optim as optim
class PPOTrainer():
    def __init__(self, num_iters : int, num_actors : int, num_timesteps : int, discount_factor : float,
                 epsilon : float, c1 : float, c2 : float, optimizer : optim.Optimizer, lr : float,
                 action_map : list, ppo_network_args : dict):
        self.pponetwork = PPONetwork(**ppo_network_args)
        self.num_iters = num_iters
        self.num_actors = num_actors
        self.num_timesteps = num_timesteps
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.action_map = action_map
        self.optimizer = optimizer(self.pponetwork.parameters(), lr = lr)

    def _compute_advantage_single_actor(self, values : tensor, rewards : tensor, firsts : list):
        """Computes advantage of a single actor"""
        T = len(rewards)
        coeffs = self.discount_factor ** torch.arange(0, T-1)
        
        ep_ends = firsts
        ep_ends.append(T-1)
        ep_idx = 0
        deltas = rewards[:-1]
        deltas = deltas + self.discount_factor * values[1:] - values[:-1]
        advantages = []
        for i in range(T-1):
            # discounts[i] = discounts[i:ep_ends[ep_idx]] * coeffs[:ep_ends[ep_idx]-i]
            discount = coeffs[:ep_ends[ep_idx] - i]
            #print(i, discount.shape, deltas[i:ep_ends[ep_idx]])
            advantages.append(torch.sum(discount * deltas[i:ep_ends[ep_idx]]))
            if i+1 == ep_ends[ep_idx]:
                deltas[i] -= self.discount_factor * values[i+1]
            if i == ep_ends[ep_idx]:
                ep_idx += 1
            
        advantages = torch.stack(advantages)
        return advantages

    def _compute_advantages(self, values : list, rewards : list, firsts : list): 
        """Computes advantage of all actors, takes values and rewards as a length N list of Tx1 tensors"""
        advantages = []
        # for vals, rews, frsts in zip(values, rewards, firsts):
        #     advantages.append(self._compute_advantage_single_actor(vals, rews, frsts))
        print(values.shape[0])
        for i in range(values.shape[0]):
            advantages.append(self._compute_advantage_single_actor(values[i], rewards[i], firsts[i]))
        return advantages

    def entropy(self, dist):
        return -torch.sum(torch.log(dist) * dist)

    def train(self, env : ProcgenGym3Env):
        obs_dims = list(env.ob_space["rgb"].shape)
        obs_dims[0] = obs_dims[-1]
        obs_dims[-1] = obs_dims[1]
        buf_dims = list(obs_dims)
        buf_dims.insert(0, self.num_actors)
        buf_dims[1] *= 3
        for i in range(self.num_iters):
            buffers = torch.zeros(buf_dims)
            firsts = [[] for _ in range(self.num_actors)]
            values = []
            alt_values = torch.zeros(self.num_actors, self.num_timesteps)
            rewards = []
            alt_rewards = torch.zeros((self.num_actors, self.num_timesteps))
            logitss = []
            for t in range(self.num_timesteps):
                rew, obs, first = env.observe()
                rew = torch.tensor(rew)
                obs = obs["rgb"] / 255.0
                obs = torch.tensor(obs)
                obs = obs.permute(0, 3, 1, 2)
                for n in range(self.num_actors):
                    buffers = torch.roll(buffers, shifts = -3, dims=1)
                    buffers[:, -3:, :, :] = obs
                    if first[n]:
                        firsts[n].append(t)
                out = self.pponetwork(buffers)
                value = out[:, 0]
                logits = F.softmax(out[:, 1:], dim=-1)
                #print("VALUE: {}, LOGITS: {}".format(value.shape, logits.shape))
                logitss.append(logits)
                act = []
                for j in range(self.num_actors):
                    act.append(torch.multinomial(logits[j], num_samples=1).detach().numpy().squeeze())
                values.append(value)
                alt_values[:, t] = value
                mapped_acts = [self.action_map[action] for action in act]
                env.act(np.array(mapped_acts))
                rewards.append(rew)
                alt_rewards[:, t] = rew
            # advantages = self._compute_advantages(values, rewards, firsts)
            advantages = self._compute_advantages(alt_values, alt_rewards, firsts)
            loss = 0
            total_sz = 0
            for t in range(self.num_timesteps-1):
                for actor in range(self.num_actors):
                    total_sz += 1
                    clamped_rewards = rewards[t][actor].clamp(min=1-self.epsilon, max=1+self.epsilon)
                    loss += torch.sum(torch.min(rewards[t][actor] * advantages[actor][t], clamped_rewards * advantages[actor][t]))
                    loss -= self.c1 * torch.sum((values[t][actor] - rewards[t][actor]) ** 2)
                    loss += self.c2 * self.entropy(logitss[t][actor])
            loss /= total_sz
            loss = -loss
            avg_reward = sum(rewards) / self.num_actors
            print("Iteration {} Training Loss {} Avg Reward {}".format(i, loss, avg_reward))
            loss.backward()
            self.optimizer.step()



                    

                    

