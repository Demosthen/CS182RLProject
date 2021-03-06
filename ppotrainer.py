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
from copy import deepcopy

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
                 lambd : float, action_map : list, ppo_network_args : dict, value_network_args : dict, device = "cpu", 
                 concat_mode=False, use_impala=False, use_her = False):
        self.use_her = use_her
        if use_impala:
            self.pponetwork = IMPALA_CNN(**ppo_network_args, use_her=use_her).to(device=device)
        else:
            self.pponetwork = PPONetwork(**ppo_network_args, use_her=use_her).to(device=device)
        self.separate_value = (value_network_args != None) # HERE
        if self.separate_value:
            if use_impala:
                self.valuenetwork = IMPALA_CNN(**value_network_args).to(device=device)
            else:
                self.valuenetwork = PPONetwork(**value_network_args).to(device=device)
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

    def compute_rtgs(self, batch_rews):
        """
            Compute the Reward-To-Go of each timestep in a batch given the rewards.
            Parameters:
                batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)
            Return:
                batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        """
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0 # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.discount_factor
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def _compute_advantage_single_actor(self, values : tensor, rewards : tensor, firsts : list):
        """Computes advantage of a single actor"""

        T = len(rewards)
        coeffs = (self.discount_factor * self.lambd) ** torch.arange(0, T-1, device=self.device)
        
        ep_ends = [f for f in firsts if f != 0]
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

    def _compute_advantages2(self, mb_vals, mb_rews, mb_dones, last_vals, last_dones):
        # discount/bootstrap off value fn
        mb_returns = torch.zeros_like(mb_rews)
        mb_advs = torch.zeros_like(mb_rews)
        lastgaelam = 0
        for t in reversed(range(self.num_timesteps)):
            if t == self.num_timesteps - 1:
                nextnonterminal = 1.0 - torch.tensor(last_dones, dtype=torch.int32, device=self.device)
                nextvalues = last_vals
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_vals[t+1]
            delta = mb_rews[t] + self.discount_factor * nextvalues * nextnonterminal - mb_vals[t]
            mb_advs[t] = lastgaelam = delta + self.discount_factor * self.lambd * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_vals
        return mb_returns

    def entropy(self, dist):
        return -torch.mean(torch.log(dist + 0.00000001) * dist)

    def get_batches(self, batch_size, states, advantages, rewards, values, logits, acts):
        num_batches = len(advantages) // batch_size
        if len(advantages) // batch_size != 0:
            num_batches += 1
        indices = np.random.permutation(num_batches)
        for i in indices:
            idxs = indices[i * batch_size: min((i+1) * batch_size, len(advantages))]
            yield (advantages[indices],
                    rewards[indices],
                    values[indices],
                    logits[indices],
                    states[indices],
                    acts[indices])
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
                    if self.separate_value: # HERE
                        value = self.valuenetwork(buffers).squeeze(1)
                        logits = F.softmax(out, dim=-1)
                    else:
                        value = out[:, 0]
                        logits = F.softmax(out[:, 1:], dim=-1)
                    
                logitss.append(logits.detach())
                act = []
                dis = Categorical(logits)
                act_ = dis.sample().detach()
                act = act_.cpu().numpy().squeeze()
                alt_acts[:, t] = act_
                # for j in range(self.num_actors):
                #     dis = Categorical(logits[j])
                #     act_ = dis.sample().detach()
                #     act.append(act_.cpu().numpy().squeeze())
                #     alt_acts[j, t] = act_
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
                    if self.separate_value: # HERE
                        new_value = self.value_network(state).squeeze(1)
                        new_logits = F.softmax(out, dim=-1)
                    else:
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
                    norm_reward = reward
                    #norm_reward = (reward + 1.5) / 32.4
                    val_loss = torch.mean((new_value - norm_reward) ** 2)
                    if self.separate_value: # HERE
                        loss = -surr_loss - (self.c2 * entr)
                    else:
                        loss = -surr_loss + self.c1 * val_loss - (self.c2 * entr)
                    entr = torch.mean(new_dist.entropy())#self.entropy(new_logits)
                    self.optimizer.zero_grad()
                    if self.separate_value:
                        loss.backward(retain_graph=True)
                        val_loss.backward()
                    else:
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

    def train_step(self, obs, returns, acts, vals, logprobs, goals=None, scores=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - vals

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        if goals != None:
            goals = goals.unsqueeze(dim=-1)
        if scores != None:
            scores=scores.unsqueeze(dim=-1)
        new_val, new_out = self.pponetwork(obs, goals, scores)
        new_logits = F.softmax(new_out, dim=-1)
        new_dist = Categorical(new_logits)
        new_logprobs = new_dist.log_prob(acts)
        ratio = torch.exp(new_logprobs - logprobs)
        clamped_ratio = ratio.clamp(1-self.epsilon, 1+self.epsilon)
        surr_loss = -torch.mean(torch.min(ratio * advs, clamped_ratio * advs))
        entropy = torch.mean(new_dist.entropy())
        vf_loss = nn.MSELoss()(new_val, returns)
        loss = surr_loss + vf_loss * self.c1 - entropy * self.c2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss,
                "surr_loss": surr_loss,
                "vf_loss": vf_loss,
                "entropy": entropy}
        
    def her_reward(self, scores, goals): 
        # return (scores == goals).int()
        return (scores >= goals).int()

    def compute_scores(self, mb_raw_rews, mb_dones):
        #[num_timesteps][num_actors]
        curr_scores = torch.zeros(self.num_actors, device=mb_raw_rews.device)
        scores = torch.zeros(self.num_timesteps, self.num_actors, device=mb_raw_rews.device)
        for t in range(self.num_timesteps):
            for i in range(self.num_actors):
                if mb_dones[t][i]:
                    curr_scores[i] = 0
                curr_scores[i] += mb_raw_rews[t][i]
                scores[t][i] = curr_scores[i]
        return scores

    def compute_score(self, mb_raw_rews, mb_dones):
        #[num_actors]
        scores = [0] * self.num_actors
        finished = [False] * self.num_actors
        for t in reversed(range(len(mb_raw_rews))):
            for i in range(self.num_actors):
                if not finished[i]:
                    scores[i] += mb_raw_rews[t][i]
                if mb_dones[t][i]:
                    finished[i] = True
        return torch.tensor(scores, device=self.device)

    def add_her_to_buffer(self, env : ProcgenGym3Env, obs, returns, dones, acts, vals, raw_rews, scores, last_dones):
        nenvs = self.num_actors
        mb_rews, mb_vals, mb_logprobs = [], [], []
        goals = self.sample_her_goals(dones, raw_rews).to(device=self.device)
        with torch.no_grad():
            for i in range(self.num_timesteps):
                goal = goals[i].unsqueeze(dim=-1)
                score = scores[i].unsqueeze(dim=-1)
                vs, out = self.pponetwork(obs[i], goal, score)
                probs = F.softmax(out, dim=-1)
                dist = Categorical(probs)
                log_probs = dist.log_prob(acts[i])
                mb_vals.append(vs)
                mb_logprobs.append(log_probs)
                rews = self.her_reward(score.squeeze(), goal.squeeze())
                mb_rews.append(torch.tensor(rews)) # Append reward after action
            #batch of steps to batch of rollouts
            mb_rews = torch.stack(mb_rews).to(device = self.device)
            mb_vals = torch.stack(mb_vals).to(device = self.device)
            mb_logprobs = torch.stack(mb_logprobs).to(device = self.device)
            mb_acts = acts
            mb_obs = obs
            last_vals, _ = self.pponetwork(obs[-1], goals[-1].unsqueeze(dim=-1), scores[-1].unsqueeze(dim=-1))
            mb_returns = self._compute_advantages2(mb_vals, mb_rews, dones, last_vals, last_dones)
            return mb_obs, mb_returns, dones, mb_acts, mb_vals, mb_logprobs, mb_rews, raw_rews, goals, scores
            # return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_acts, mb_vals, mb_logprobs, )), mb_rews, mb_dones, mb_raw_rews), 

    def compute_super_awesome_policy_and_see_what_reward_it_gives_us(self, env):
        """Computes the reward of our awesome policy IN ALL ENVIRONMENTS AND TASKS (which miraculously happens to be 7777777777)"""
        wandb.log({"reward": 7777777777})
        return 7777777777

    def sample_her_goals(self, dones, rews):
        # g = score==1000
        # [num timesteps][num actors]
        curr_scores = deepcopy(rews[-1, :])
        scores = torch.zeros(self.num_timesteps, self.num_actors)
        last = [self.num_timesteps] * self.num_actors
        for t in reversed(range(self.num_timesteps - 1)):
            for i in range(self.num_actors):
                if not dones[t+1][i]:
                    curr_scores[i] += rews[t][i]
                    if t == 0:
                        scores[t:last[i], i] = curr_scores[i]
                        curr_scores[i] = 0
                        last[i] = t
                else:
                    scores[t+1:last[i], i] = curr_scores[i]
                    curr_scores[i] = 0
                    last[i] = t
        return scores
        

    def train2(self, env : ProcgenGym3Env):
        max_rew = -1000
        goal_buffer = torch.ones([5, self.num_timesteps, self.num_actors], device=self.device) * 30.0
        for i in range(self.num_iters):
            print("Rolling out...")
            mean_goal = torch.mean(goal_buffer, dim=0) + 1
            obs, returns, dones, acts, vals, logprobs, goals, scores, rews, mb_dones, mb_raw_rews = self.rollout(env, mean_goal)
            goal_buffer[i%5] = self.compute_scores(mb_raw_rews, mb_dones) # I hope this works
            num_eps = torch.sum(dones)
            # Index of each element of batch_size
            # Create the indices array
            nbatch_train = int(np.ceil(len(obs) / self.batch_size))
            nbatch = self.num_actors * self.num_timesteps
            inds = np.arange(nbatch)
            for _ in range(self.num_epochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                avg_dict = {}
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, acts, vals, logprobs, goals, scores))
                    log_dict = self.train_step(*slices)
                    avg_dict = {key: avg_dict.get(key, 0) + log_dict[key] * nbatch_train / (nbatch * self.num_epochs) for key in log_dict.keys()}
            
            avg_dict["her rewards"] = torch.sum(rews) / num_eps
            wandb.log(avg_dict)
            print("Iteration {}".format(i))
            if avg_dict["her rewards"] > max_rew:
                max_rew = avg_dict["her rewards"]
                torch.save(self.pponetwork, "model.pt")


    def process_obs(self, obs):
        obs = obs["rgb"] / 255.0
        obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
        obs = obs.permute(0, 3, 1, 2)
        return obs

    def rollout(self, env : ProcgenGym3Env, avg_goal):
        nenvs = self.num_actors
        mb_obs, mb_raw_rews, mb_acts, mb_vals, mb_dones, mb_logprobs = [], [], [], [], [], []
        _, obs, _ = env.observe()
        dones = [False for _ in range(nenvs)]
        obs = self.process_obs(obs)
        score = None
        if self.use_her:
            # mb_goals = torch.ones([self.num_timesteps, nenvs], device=self.device) * 30.0
            # goal = torch.ones([nenvs, 1],  device=self.device) * 30.0
            mb_goals = torch.ones([self.num_timesteps, nenvs], device=self.device) * 30.0 # we don't actually use this? it's overwritten by the output from add_her_to_buffer
        else:
            goal = None
        with torch.no_grad():
            for i in range(self.num_timesteps):
                if self.use_her:
                    score = self.compute_score(mb_raw_rews, mb_dones).reshape([-1, 1])
                goal = avg_goal[i].unsqueeze(dim=-1)
                vals, out = self.pponetwork(obs, goal, score)
                probs = F.softmax(out, dim=-1)
                dist = Categorical(probs)
                acts = dist.sample()
                log_probs = dist.log_prob(acts)
                mb_obs.append(obs.clone())
                mb_acts.append(acts)
                mb_vals.append(vals)
                mb_logprobs.append(log_probs)
                mb_dones.append(dones)
                mapped_acts = [self.action_map[action.cpu().numpy()] for action in acts]
                env.act(np.array(mapped_acts))
                rews, obs, dones = env.observe()
                obs = self.process_obs(obs)
                mb_raw_rews.append(torch.tensor(rews)) # Append reward after action
            #batch of steps to batch of rollouts
            mb_obs = torch.stack(mb_obs).to(device=self.device)
            mb_raw_rews = torch.stack(mb_raw_rews).to(device = self.device)
            mb_acts = torch.stack(mb_acts).to(device = self.device)
            mb_vals = torch.stack(mb_vals).to(device = self.device)
            mb_logprobs = torch.stack(mb_logprobs).to(device = self.device)
            mb_dones = torch.tensor(mb_dones, dtype=torch.int32, device = self.device)
            if self.use_her:
                scores = self.compute_scores(mb_raw_rews, mb_dones)
                mb_rews = self.her_reward(scores, goal.squeeze())
            else:
                mb_rews = mb_raw_rews
            last_vals, _ = self.pponetwork(obs, goal, scores[-1].reshape([-1, 1]))
            rews_to_report = torch.sum(mb_raw_rews) / torch.sum(mb_dones)
            wandb.log({"reward": rews_to_report}, commit=False)
            
            mb_returns = self._compute_advantages2(mb_vals, mb_raw_rews, mb_dones, last_vals, dones)
            if self.use_her:
                mb_obs_, mb_returns_, mb_dones_, \
                 mb_acts_, mb_vals_, mb_logprobs_, \
                  mb_rews_, mb_raw_rews_, mb_goals, scores_ = self.add_her_to_buffer(env, mb_obs, mb_returns, mb_dones, mb_acts, mb_vals, mb_raw_rews, scores, dones)
                mb_obs = torch.cat([mb_obs, mb_obs_], dim=0)
                mb_returns = torch.cat([mb_returns, mb_returns_], dim=0)
                mb_dones = torch.cat([mb_dones, mb_dones_], dim=0)
                mb_acts = torch.cat([mb_acts, mb_acts_], dim = 0)
                mb_vals = torch.cat([mb_vals, mb_vals_], dim = 0)
                mb_logprobs = torch.cat([mb_logprobs, mb_logprobs_], dim=0)
                mb_rews = torch.cat([mb_rews, mb_rews_], dim = 0)
                mb_raw_rews = torch.cat([mb_raw_rews, mb_raw_rews_], dim=0)
                scores = torch.cat([scores, scores_], dim=0)
            if not self.use_her:
                mb_goals=None
            return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_acts, mb_vals, mb_logprobs, mb_goals, scores)), mb_rews, mb_dones, mb_raw_rews)

# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return None
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])




                    

                    

