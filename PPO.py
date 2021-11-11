# %%
import numpy as np
import gym
import torch
from torch import nn
from torch.distributions.normal import Normal
import torch.nn.functional as F

from tqdm.std import tqdm
import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# %%
env = gym.make('Pendulum-v0')
n_state = int(np.prod(env.observation_space.shape))
n_action = int(np.prod(env.action_space.shape))
print("# of state", n_state)
print("# of action", n_action)

# %%


def run_episode(env, policy, render=False):
    obs_list = []
    act_list = []
    reward_list = []
    next_obs_list = []
    done_list = []
    obs = env.reset()
    while True:
        if render:
            env.render()

        action = policy(obs)
        next_obs, reward, done, _ = env.step(action)
        reward_list.append(reward), obs_list.append(obs), \
            done_list.append(done), act_list.append(action), \
            next_obs_list.append(next_obs)
        if done:
            break
        obs = next_obs

    return obs_list, act_list, reward_list, next_obs_list, done_list

# %%


class PPO():
    def __init__(self, n_state, n_action):
        # Define network
        self.act_net = nn.Sequential(
            nn.Linear(n_state, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2*n_action),
        )
        self.act_net.to(device)
        self.old_act_net = copy.deepcopy(self.act_net)
        self.old_act_net.to(device)
        self.v_net = nn.Sequential(
            nn.Linear(n_state, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.v_net.to(device)
        self.v_optimizer = torch.optim.Adam(self.v_net.parameters(), lr=1e-3)
        self.act_optimizer = torch.optim.Adam(
            self.act_net.parameters(), lr=1e-4)
        self.old_v_net = copy.deepcopy(self.v_net)
        self.old_v_net.to(device)
        self.gamma = 0.95
        self.gae_lambda = 0.85
        self._eps_clip = 0.2
        self.act_lim = 2

    def __call__(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            # calculate old logprob
            output = self.act_net(state)
            mu = self.act_lim*torch.tanh(output[:n_action])
            var = torch.abs(output[n_action:])
            dist = Normal(mu, var)
            action = dist.sample()
            action = action.detach().cpu().numpy()
        return np.clip(action, -self.act_lim, self.act_lim)

    def update(self, data=None):
        obs, act, reward, next_obs, done = data
        # Calculate culmulative return
        obs = torch.FloatTensor(obs).to(device)
        next_obs = torch.FloatTensor(next_obs).to(device)
        act = torch.FloatTensor(act).to(device)
        with torch.no_grad():
            v_s = self.old_v_net(obs).detach().cpu().numpy().squeeze()
            v_s_ = self.old_v_net(next_obs).detach().cpu().numpy().squeeze()
            output = self.old_act_net(obs)
            mu = self.act_lim*torch.tanh(output[:, :n_action])
            var = torch.abs(output[:, n_action:2*n_action])
            dist = Normal(mu, var)
            old_logprob = dist.log_prob(act)

        adv = np.zeros_like(reward)
        done = np.array(done, dtype=float)

        returns = np.zeros_like(reward)
        # # One-step
        # adv = reward + (1-done)*self.gamma*v_s_ - v_s
        # returns = adv + v_s
        # MC
        # s = 0
        # for i in reversed(range(len(returns))):
        #     s = s * self.gamma + reward[i]
        #     returns[i] = s
        # adv = returns - v_s
        # # GAE
        delta = reward + v_s_ * self.gamma - v_s
        m = (1.0 - done) * (self.gamma * self.gae_lambda)
        gae = 0.0
        for i in range(len(reward) - 1, -1, -1):
            gae = delta[i] + m[i] * gae
            adv[i] = gae
        returns = adv + v_s

        adv = torch.FloatTensor(adv).to(device)
        returns = torch.FloatTensor(returns).to(device)
        # Calculate loss
        batch_size = 32
        list = [j for j in range(len(obs))]
        for i in range(0, len(list), batch_size):
            index = list[i:i+batch_size]
            for _ in range(1):
                output = self.act_net(obs[index])
                mu = self.act_lim*torch.tanh(output[:, :n_action])
                var = torch.abs(output[:, n_action:])
                dist = Normal(mu, var)
                logprob = dist.log_prob(act[index])

                ratio = (logprob - old_logprob[index]).exp().float().squeeze()
                surr1 = ratio * adv[index]
                surr2 = ratio.clamp(1.0 - self._eps_clip, 1.0 +
                                    self._eps_clip) * adv[index]
                act_loss = -torch.min(surr1, surr2).mean()

                ent_loss = dist.entropy().mean()
                act_loss -= 0.01*ent_loss
                self.act_optimizer.zero_grad()
                act_loss.backward()
                self.act_optimizer.step()

            for _ in range(1):
                v_loss = F.mse_loss(self.v_net(
                    obs[index]).squeeze(), returns[index])
                self.v_optimizer.zero_grad()
                v_loss.backward()
                self.v_optimizer.step()

        return act_loss.item(), v_loss.item(), ent_loss.item()


# %%
loss_act_list, loss_v_list, loss_ent_list, reward_list = [], [], [], []
agent = PPO(n_state, n_action)
loss_act, loss_v = 0, 0
n_step = 0
for i in tqdm(range(3000)):
    data = run_episode(env, agent)
    agent.old_v_net.load_state_dict(agent.v_net.state_dict())
    agent.old_act_net.load_state_dict(agent.act_net.state_dict())
    for _ in range(2):
        loss_act, loss_v, loss_ent = agent.update(data)
    rew = sum(data[2])
    if i > 0 and i % 50 == 0:
        run_episode(env, agent, True)[2]
        print("itr:({:>5d}) loss_act:{:>6.4f} loss_v:{:>6.4f} loss_ent:{:>6.4f} reward:{:>3.1f}".format(i, np.mean(
            loss_act_list[-50:]), np.mean(loss_v_list[-50:]),
            np.mean(loss_ent_list[-50:]), np.mean(reward_list[-50:])))

    loss_act_list.append(loss_act), loss_v_list.append(
        loss_v), loss_ent_list.append(loss_ent), reward_list.append(rew)

# %%
scores = [sum(run_episode(env, agent, False)[2]) for _ in range(100)]
print("Final score:", np.mean(scores))

import pandas as pd
df = pd.DataFrame({'loss_v': loss_v_list,
                   'loss_act': loss_act_list,
                   'loss_ent': loss_ent_list,
                   'reward': reward_list})
df.to_csv("./ClassMaterials/Lecture_21_PPO/data/ppo.csv",
          index=False, header=True)
