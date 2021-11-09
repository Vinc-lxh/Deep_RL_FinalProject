# %%
import numpy as np
import gym
import torch
from torch import nn
from torch.distributions.normal import Normal
import torch.nn.functional as F
from collections import deque
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

class Policy(nn.Module):
    def __init__(self, n_state, n_action):
        super(Policy, self).__init__()
        self.hidden = 256

        self.fc = nn.Sequential(
            nn.Linear(n_state, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.theta_mu_head = nn.Linear(128, 2*n_action)
        self.v_aux_head = nn.Linear(128, 1)

    def forward(self, state):
        out = self.fc(state)
        theta_mu = self.theta_mu_head(out)
        v_aux = self.v_aux_head(out)

        return theta_mu, v_aux


class Value(nn.Module):
    def __init__(self, n_state, n_action):
        super(Value, self).__init__()
        self.hidden = 256

        self.fc = nn.Sequential(
            nn.Linear(n_state, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        #self.v = nn.Linear(128, 1)

    def forward(self, state):
        v_out = self.fc(state)
        #out = self.v(out)
        return v_out


def kl(mu1,var1,mu2,var2):
    return np.log(var2/var1)+(var1**2+(mu1-mu2)**2)/(2*(var2**2))-1/2


class PPG():
    def __init__(self, n_state, n_action):
        # Define network
        self.policy_net = Policy(n_state,n_action) #object of Policy_network
        self.policy_net.to(device)
        self.old_policy_net = copy.deepcopy(self.policy_net)
        self.old_policy_net.to(device)


        self.v_net = Value(n_state,n_action)        #object of Value_network
        self.v_net.to(device)
        self.old_v_net = copy.deepcopy(self.v_net)
        self.old_v_net.to(device)

        self.v_optimizer = torch.optim.Adam(self.v_net.parameters(), lr=1e-4)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-4)
        

        self.gamma = 0.95
        self.gae_lambda = 0.85
        self._eps_clip = 0.2
        self.act_lim = 2
        self.beta_clone = 1 #beta clone to control the entrogy bonus
        self.E_Pi = 1  #number of iteration of Policy Epochs
        self.E_v = 1   #number of iteration of Value Epochs
        self.E_aux = 6 #numer of iteration of Auxiliary Phase
        self.buff = Buffer()

    def __call__(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            # calculate old logprob
            output,_ = self.policy_net(state)
            mu = self.act_lim*torch.tanh(output[:n_action])
            var = torch.abs(output[n_action:])
            dist = Normal(mu, var)
            action = dist.sample()
            action = action.detach().cpu().numpy()
        return np.clip(action, -self.act_lim, self.act_lim)

    def update_aux(self, data=None):
        _, act, reward, next_obs, done = data
        
        obs, V_targ, old_mu,old_theta = self.buff.data() #obs = obs
        # Calculate culmulative return
        obs = torch.FloatTensor(obs).to(device)
        next_obs = torch.FloatTensor(next_obs).to(device)
        act = torch.FloatTensor(act).to(device)
        V_targ = torch.FloatTensor(V_targ).to(device)
        old_mu = torch.FloatTensor(old_mu).to(device).squeeze()
        old_theta = torch.FloatTensor(old_theta).to(device).squeeze()

        batch_size = 32
        list = [j for j in range(len(obs))]
        for i in range(0, len(list), batch_size):
            index = list[i:i+batch_size]
            for _ in range(self.E_aux):
                output,v_aux = self.policy_net(obs[index])  #extract theta and mu, v_aux from the network
                mu = self.act_lim*torch.tanh(output[:, :n_action]).detach()
                var = torch.abs(output[:, n_action:]).detach()
                # dist = Normal(mu, var)
            
                aux_loss =  F.mse_loss(v_aux.squeeze(), V_targ[index])
                #new_log_prob = dist.log_prob(act[index]).squeeze()
                #loss_kl = F.kl_div(new_log_prob,old_log_prob[index],reduction='batchmean')
                loss_kl = kl(mu,var,old_mu,old_theta)
                policy_loss = aux_loss + self.beta_clone*loss_kl
                self.policy_optimizer.zero_grad()
                policy_loss.sum().backward()
                self.policy_optimizer.step()

                v_loss = F.mse_loss(self.v_net(obs[index]).squeeze(), V_targ[index])
                self.v_optimizer.zero_grad()
                v_loss.backward()
                self.v_optimizer.step()
        
        self.buff.clean()

        return policy_loss.sum().item(), v_loss.item()

    def update(self, data=None):
        obs, act, reward, next_obs, done = data
        # Calculate culmulative return
        obs = torch.FloatTensor(obs).to(device)
        next_obs = torch.FloatTensor(next_obs).to(device)
        act = torch.FloatTensor(act).to(device)
        with torch.no_grad():
            v_s = self.old_v_net(obs).detach().cpu().numpy().squeeze()
            v_s_ = self.old_v_net(next_obs).detach().cpu().numpy().squeeze()
            output,_ = self.old_policy_net(obs)
            mu = self.act_lim*torch.tanh(output[:, :n_action])
            var = torch.abs(output[:, n_action:])
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
            for _ in range(self.E_Pi):
                output,_ = self.policy_net(obs[index])  #?
                mu = self.act_lim*torch.tanh(output[:, :n_action])
                var = torch.abs(output[:, n_action:])
                dist = Normal(mu, var)
                logprob = dist.log_prob(act[index])

                ratio = (logprob - old_logprob[index]).exp().float().squeeze()
                surr1 = ratio * adv[index]
                surr2 = ratio.clamp(1.0 - self._eps_clip, 1.0 +
                                    self._eps_clip) * adv[index]
                act_loss = -torch.min(surr1, surr2).mean()

                ent_loss = dist.entropy().mean()            #entroy loss
                act_loss -= 0.01 * ent_loss      #L_clip - beta_clone * entropy_bonus
                self.policy_optimizer.zero_grad()
                act_loss.backward()
                self.policy_optimizer.step()

            self.buff.add(obs[index], returns[index], mu.detach(),var.detach()) #State, V_targ, logprob, ask if that make sense?

            for _ in range(self.E_v):
                v_loss = F.mse_loss(self.v_net(
                    obs[index]).squeeze(), returns[index])
                self.v_optimizer.zero_grad()
                v_loss.backward()
                self.v_optimizer.step()

            
        return act_loss.item(), v_loss.item(), ent_loss.item()


class Buffer:
    def __init__(self):
        self.obs_list = []
        self.V_targ_list = []
        self.mu_list = []
        self.theta_list = []

    def add(self,obs, V_targ, mu,theta):
        self.obs_list.append(obs)
        self.V_targ_list.append(V_targ)
        self.mu_list.append(mu)
        self.theta_list.append(theta)

    def data(self):
        return torch.cat(self.obs_list), torch.cat(self.V_targ_list), torch.cat(self.mu_list),torch.cat(self.theta_list)
    
    def clean(self):
        self.obs_list = []
        self.V_targ_list = []
        self.old_log_prob_list = []

    def __len__(self):
        return len(self.buff)


        
        
# %%
loss_act_list, loss_v_list, loss_ent_list, reward_list,aux_act_list,aux_v_list = [], [], [], [],[],[]
agent = PPG(n_state, n_action)
loss_act, loss_v, aux_act,aux_v = 0, 0, 0, 0
n_step = 0
for i in tqdm(range(3000)):
    data = run_episode(env, agent)
    agent.old_v_net.load_state_dict(agent.v_net.state_dict())
    agent.old_policy_net.load_state_dict(agent.policy_net.state_dict())
    
    loss_act, loss_v, loss_ent = agent.update(data)
    aux_act,aux_v = agent.update_aux(data)

    rew = sum(data[2])
    if i > 0 and i % 50 == 0:
        run_episode(env, agent, True)[2]
        print("\nitr:({:>5d}) loss_act:{:>6.4f} loss_v:{:>6.4f} aux_act:{:>6.4f} aux_v:{:>6.4f}  loss_ent:{:>6.4f} reward:{:>3.1f}".format(i, np.mean(
            loss_act_list[-50:]), np.mean(loss_v_list[-50:]), np.mean(aux_act_list[-50:]),np.mean(aux_v_list[-50:]),
            np.mean(loss_ent_list[-50:]), np.mean(reward_list[-50:])))

    loss_act_list.append(loss_act), loss_v_list.append(loss_v), loss_ent_list.append(loss_ent), reward_list.append(rew), aux_act_list.append(aux_act), aux_v_list.append(aux_v)

# %%
scores = [sum(run_episode(env, agent, False)[2]) for _ in range(100)]
print("Final score:", np.mean(scores))

import pandas as pd
df = pd.DataFrame({'loss_v': loss_v_list,
                   'loss_act': loss_act_list,
                   'loss_ent': loss_ent_list,
                   'aux_act': aux_act_list,
                   'aux_v': aux_v_list,
                   'reward': reward_list})
df.to_csv("./ClassMaterials/Lecture_21_PPG/data/PPG.csv",
          index=False, header=True)
