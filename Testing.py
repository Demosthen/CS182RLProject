import gym3
from procgen import ProcgenGym3Env
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ppotrainer import *
from pponetwork import * 
env = ProcgenGym3Env(num=1, env_name="fruitbot", render_mode="rgb_array")
env = gym3.ViewerWrapper(env, info_key="rgb")
step = 0
# TODO: decrease action space to just 3 
# [
#             ("LEFT", "DOWN"),
#             ("LEFT",),
#             ("LEFT", "UP"),
#             ("DOWN",),
#             (),
#             ("UP",),
#             ("RIGHT", "DOWN"),
#             ("RIGHT",),
#             ("RIGHT", "UP"),
#             ("D",),
#             ("A",),
#             ("W",),
#             ("S",),
#             ("Q",),
#             ("E",),
#         ]
# while True:
#     act = gym3.types_np.sample(env.ac_space, bshape=(env.num,))
#     env.act(act)
#     rew, obs, first = env.observe()
#     print(f"step {step} reward {rew} first {first}, action {act}")
#     step += 1
action_map = [1, 4, 7, 9]
network_params = {'input_dims': (64, 64, 9),
                    'num_actions': 4,
                    'conv_layer_sizes': [32, 64, 64],
                    'fc_layer_sizes': [512],
                    'strides': [4, 2, 1],
                    }
trainer = PPOTrainer(num_iters = 1000,
                     num_actors = 1,
                     num_timesteps = 16, 
                     discount_factor = 0.99,
                     epsilon = 0.2,
                     c1 = 0.01,
                     c2 = 0.1,
                     optimizer = torch.optim.Adam,
                     lr= 3e-4,
                     action_map=action_map,
                     ppo_network_args = network_params)
trainer.train(env)
