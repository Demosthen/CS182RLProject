import gym3
from procgen import ProcgenGym3Env
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ppotrainer import *
from pponetwork import * 
import wandb
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
num_actors = 64
env = ProcgenGym3Env(num=num_actors, env_name="fruitbot", render_mode="rgb_array", distribution_mode="easy")
#env = gym3.ViewerWrapper(env, info_key="rgb")
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
concat_mode = False
impala_params = {"in_channels": 3,
                 "depths": [16, 32, 32],
                 "out_dim": 4}
a_impala_params = {"in_channels": 3,
                 "depths": [16, 32, 32],
                 "out_dim": 3}
v_impala_params = {"in_channels": 3,
                 "depths": [16, 32, 32],
                 "out_dim": 0}
cnn_params = {'input_dims': (64, 64, 9) if concat_mode else (64, 64, 3),
                    'num_actions': 4,
                    'conv_layer_sizes': [],
                    'fc_layer_sizes': [64, 64],
                    'strides': [4, 2],
                    'filter_sizes': [8, 4]
                    }
a_cnn_params = {'input_dims': (64, 64, 9) if concat_mode else (64, 64, 3),
                    'num_actions': 3,
                    'conv_layer_sizes': [8, 16],
                    'fc_layer_sizes': [128],
                    'strides': [4, 2],
                    'filter_sizes': [8, 4]
                    }
v_cnn_params = {'input_dims': (64, 64, 9) if concat_mode else (64, 64, 3),
                    'num_actions': 0,
                    'conv_layer_sizes': [8, 16],
                    'fc_layer_sizes': [128],
                    'strides': [4, 2],
                    'filter_sizes': [8, 4]
                    }
use_impala = True
if use_impala:
    network_params = impala_params
else:
    network_params = cnn_params
separate_value = True
if separate_value:
    if use_impala:
        network_params = a_impala_params
        value_params = v_impala_params
    else:
        network_params = a_cnn_params
        value_params = v_cnn_params
# wandb.init(project="cs182rlproject")
trainer = PPOTrainer(num_iters = 5000,
                     num_actors = num_actors,
                     num_timesteps = 256, 
                     discount_factor = 0.999,
                     epsilon = 0.2,
                     c1 = 0.1,
                     c2 = 0.01,
                     optimizer = torch.optim.Adam,
                     lr= 5e-4,
                     action_map=action_map,
                     lambd=0.95,
                     num_epochs=3,
                     batch_size = 256*num_actors // 8,
                     concat_mode=concat_mode,
                     device=device,
                     use_impala=use_impala,
                     ppo_network_args = network_params,
                     value_network_args = value_params)

#print(trainer._compute_advantage_single_actor(torch.ones([10]), torch.ones([10]), [2, 5]))
trainer.train2(env)
