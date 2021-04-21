import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from pponetwork import IMPALA_CNN, PPONetwork
import gym3
from procgen import ProcgenGym3Env
import numpy as np
import torch.optim as optim
import wandb
from torch.distributions import Categorical

def explained_variance(ypred,y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    """
    vary = torch.var(y)
    return np.nan if vary==0 else 1 - torch.var(y-ypred)/vary

class PPOTrainer():
    def __init__(self, num_iters : int, num_actors : int, num_timesteps : int, num_epochs : int, discount_factor : float,
                 batch_size : int, epsilon : float, c1 : float, c2 : float, optimizer : optim.Optimizer, lr : float, 
                 lambd : float, action_map : list, ppo_network_args : dict, device = "cpu", concat_mode=False, use_impala=False):
        if use_impala:
            self.pponetwork = IMPALA_CNN(**ppo_network_args).to(device=device)
        else:
            self.pponetwork = PPONetwork(**ppo_network_args).to(device=device)
        wandb.watch(self.pponetwork)
        self.num_iters = num_iters
        self.num_actors = num_actors
        self.num_timesteps = num_timesteps
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.action_map = action_map
        self.optimizer = optimizer(self.pponetwork.parameters(), lr = lr)
        self.device = device
        self.lambd = lambd
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.concat_mode = concat_mode

    def _compute_advantage_single_actor(self, values : tensor, rewards : tensor, firsts : list):
        """Computes advantage of a single actor"""
        T = len(rewards)
        coeffs = (self.discount_factor * self.lambd) ** torch.arange(0, T-1, device=self.device)
        
        ep_ends = firsts
        ep_ends.append(T-1)
        ep_idx = 0
        deltas = rewards[:-1]
        deltas = deltas + self.discount_factor * values[1:] - values[:-1]
        advantages = []
        for i in range(T-1):
            # discounts[i] = discounts[i:ep_ends[ep_idx]] * coeffs[:ep_ends[ep_idx]-i]
            discount = coeffs[:ep_ends[ep_idx] - i]
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
        for i in range(values.shape[0]):
            advantages.append(self._compute_advantage_single_actor(values[i], rewards[i], firsts[i]))
        advantages = torch.stack(advantages)
        return (advantages - torch.mean(advantages)) / (torch.std(advantages) + 0.00001)

    def entropy(self, dist):
        return -torch.mean(torch.log(dist + 0.00000001) * dist)

    def get_batches(self, batch_size, states, advantages, rewards, values, logits, acts):
        num_batches = len(advantages) // batch_size
        if len(advantages) // batch_size != 0:
            num_batches += 1
        indices = np.random.permutation(num_batches)
        for i in indices:
            yield (advantages[i * batch_size: min((i+1) * batch_size, len(advantages))],
                    rewards[i * batch_size: min((i+1) * batch_size, len(rewards))],
                    values[i * batch_size: min((i+1) * batch_size, len(values))],
                    logits[i * batch_size: min((i+1) * batch_size, len(logits))],
                    states[i * batch_size: min((i+1) * batch_size, len(states))],
                    acts[i * batch_size: min((i+1) * batch_size, len(acts))])

    def train(self, env : ProcgenGym3Env):
        self.pponetwork.train()
        obs_dims = list(env.ob_space["rgb"].shape)
        obs_dims[0] = obs_dims[-1]
        obs_dims[-1] = obs_dims[1]
        buf_dims = list(obs_dims)
        buf_dims.insert(0, self.num_actors)
        if self.concat_mode:
            buf_dims[1] *= 3
        best_reward = -10000
        for i in range(self.num_iters):
            buffers = torch.zeros(buf_dims, device=self.device)
            firsts = [[] for _ in range(self.num_actors)]
            values = []
            alt_values = torch.zeros(self.num_actors, self.num_timesteps, device=self.device)
            rewards = []
            alt_rewards = torch.zeros((self.num_actors, self.num_timesteps), device=self.device)
            logitss = []
            if self.concat_mode:
                states = torch.zeros((self.num_actors * self.num_timesteps, 9, 64, 64), device=self.device)
            else:
                states = torch.zeros((self.num_actors * self.num_timesteps, 3, 64, 64), device=self.device)
            alt_acts = torch.zeros((self.num_actors, self.num_timesteps), device=self.device, dtype=torch.long)
            state_ctr = 0
            for t in range(self.num_timesteps):
                rew, obs, first = env.observe()
                rew = torch.tensor(rew, device=self.device, dtype=torch.float32)
                # rew += 1.5
                # rew /= 32.4
                obs = obs["rgb"] / 255.0
                obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
                obs = obs.permute(0, 3, 1, 2)
                for n in range(self.num_actors):
                    if self.concat_mode:
                        buffers = torch.roll(buffers, shifts = -3, dims=1)
                        buffers[:, -3:, :, :] = obs
                    else:
                        buffers = obs
                    if first[n]:
                        firsts[n].append(t)
                states[state_ctr: state_ctr + len(buffers)] = buffers
                with torch.no_grad():
                    out = self.pponetwork(buffers)
                    value = out[:, 0]
                    logits = F.softmax(out[:, 1:], dim=-1)
                logitss.append(logits.detach())
                act = []
                for j in range(self.num_actors):
                    dis = Categorical(logits[j])
                    act_ = dis.sample().detach()
                    act.append(act_.cpu().numpy().squeeze())
                    alt_acts[j, t] = act_
                values.append(value.detach())
                alt_values[:, t] = value
                
                mapped_acts = [self.action_map[action] for action in act]
                env.act(np.array(mapped_acts))
                rewards.append(rew)
                alt_rewards[:, t] = rew
                state_ctr += len(buffers)
            # advantages = self._compute_advantages(values, rewards, firsts)
            advantages = self._compute_advantages(alt_values, alt_rewards, firsts)
            loss = 0
            avg_reward = alt_rewards.sum(dim=-1).mean()
            advantages = torch.flatten(advantages, start_dim=0, end_dim=1).detach()
            alt_rewards = torch.flatten(alt_rewards, start_dim=0, end_dim=1)[:-self.num_actors]
            alt_values = torch.flatten(alt_values, start_dim=0, end_dim=1)[:-self.num_actors]
            alt_acts = torch.flatten(alt_acts, start_dim=0, end_dim=1)[:-self.num_actors]
            logitss = torch.flatten(torch.stack(logitss), start_dim=0, end_dim=1)[:-self.num_actors] # I hope that works
            states = states[:-self.num_actors]
            total_sz = len(advantages) * self.num_epochs
            avg_loss = 0
            #avg_reward = 0
            avg_entropy = 0
            avg_val_loss = 0
            avg_surr_loss = 0
            ev_avg = 0
            for epoch in range(self.num_epochs):
                data_gen = self.get_batches(self.batch_size, states, advantages, alt_rewards, alt_values, logitss, alt_acts)
                for advantage, reward, value, logit, state, act in data_gen:
                    out = self.pponetwork(state) # N x (1 + action_space)
                    new_value = out[:, 0]
                    new_logits = F.softmax(out[:, 1:], dim=-1)
                    new_dist = Categorical(new_logits)
                    old_dist = Categorical(logit)
                    new_log_probs = new_dist.log_prob(act)
                    old_log_probs = old_dist.log_prob(act)
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    #ratio = torch.stack([ratio[m, act[m]] for m in range(len(ratio))])# ratio[torch.arange(len(ratio)), act]
                    
                    #advantage = advantage.view([-1, 1])
                    clamped_reward = ratio.clamp(min=1-self.epsilon, max=1+self.epsilon)
                    surr_loss = torch.mean(torch.min(ratio * advantage, clamped_reward * advantage))
                    norm_reward = (reward + 1.5) / 32.4
                    val_loss = torch.mean((new_value - norm_reward) ** 2)
                    entr = self.entropy(new_logits)
                    loss = -surr_loss + self.c1 * val_loss - (self.c2 * entr)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    batch_sz = len(advantage)
                    avg_loss += loss * batch_sz / total_sz
                    ev = explained_variance(new_value, norm_reward)
                    ev_avg += ev * batch_sz / total_sz
                    #avg_reward += torch.sum(reward) / total_sz
                    avg_entropy += entr / total_sz
                    avg_val_loss += val_loss * batch_sz / total_sz
                    avg_surr_loss += surr_loss * batch_sz / total_sz
            print("Iteration {} Training Loss {:2f} Avg Reward {:2f} Avg Entropy {:2f} Avg Val_loss {:2f} Avg surr gain: {:2f} Avg exp variance: {:2f}".format(i, avg_loss, avg_reward, avg_entropy, avg_val_loss, avg_surr_loss, ev_avg))
            
            wandb.log({"Training loss": avg_loss,
                     "Avg reward": avg_reward,
                     "Avg entropy": avg_entropy,
                     "Avg val loss": avg_val_loss,
                     "Avg surr loss": avg_surr_loss,
                     "Avg explained variance": ev_avg})
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(self.pponetwork, "model.pt")



                    

                    

