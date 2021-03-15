import torch
import torch.nn as nn
import torch.nn.functional as F
from pponetwork import PPONetwork
import gym3
from procgen import ProcgenGym3Env
class PPOTrainer():
    def __init__(self, num_iters : int, num_actors : int, num_timesteps : int, ppo_network_args : dict):
        self.pponetwork = PPONetwork(**ppo_network_args)
        self.num_iters = num_iters
        self.num_actors = num_actors
        self.num_timesteps = num_timesteps

    def train(self, env : ProcgenGym3Env):
        obs_dims = env.ob_space["rgb"].shape
        buf_dims = list(obs_dims)
        buf_dims.insert(0, self.num_actors)
        buf_dims[-1] *= 3
        for i in range(self.num_iters):
            buffers = torch.zeros(buf_dims)
            firsts = [[] for _ in range(self.num_actors)]
            values = []
            for t in range(self.num_timesteps):
                rew, obs, first = env.observe()
                for n in range(self.num_actors):
                    buffers = torch.roll(buffers, shifts = -3, dims=-1)
                    buffers[:, :, :, -3:] = obs
                    if first[n]:
                        firsts[n].append(t)
                out = self.pponetwork(buffers)
                value = out[:, 0]
                act = torch.argmax(out[:, 1:], dim=-1)
                values.append(value)
                env.act(act - 1)
            # TODO:  do advantage stuff and train stuff



                    

                    

