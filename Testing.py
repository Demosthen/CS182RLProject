import gym3
from procgen import ProcgenGym3Env
import numpy as np
env = ProcgenGym3Env(num=2, env_name="fruitbot", render_mode="rgb_array")
env = gym3.ViewerWrapper(env, info_key="rgb")
step = 0
print(env.ob_space["rgb"].shape)
# TODO: decrease action space to just 3 
[
            ("LEFT", "DOWN"),
            ("LEFT",),
            ("LEFT", "UP"),
            ("DOWN",),
            (),
            ("UP",),
            ("RIGHT", "DOWN"),
            ("RIGHT",),
            ("RIGHT", "UP"),
            ("D",),
            ("A",),
            ("W",),
            ("S",),
            ("Q",),
            ("E",),
        ]
while True:
    act = gym3.types_np.sample(env.ac_space, bshape=(env.num,))
    env.act(act)
    rew, obs, first = env.observe()
    print(f"step {step} reward {rew} first {first}, action {act}")
    step += 1