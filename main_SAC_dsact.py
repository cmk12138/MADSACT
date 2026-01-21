# -*- coding: utf-8 -*-
# 开发者：Cmk
# 开发时间：2025/12/7 18:13
# NOTE: This copy is dedicated to replacing the SAC update rules with DSAC-T style
#       updates while preserving the original path-planning environment workflow.
from rl_env.path_env import RlGame
from torch.optim.lr_scheduler import CosineAnnealingLR
# import pygame
# from assignment import constants as C
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle as pkl
shoplistfile = 'E:/PycharmProjects/code/backup/path planning/DSACT_new1'  #保存文件数据所在文件的文件名
shoplistfile_test = 'E:/PycharmProjects/code/backup/path planning/DSACT_d_test2'  #保存文件数据所在文件的文件名
shoplistfile_test1 = 'E:/PycharmProjects/code/backup/path planning/DSACT_compare'  #保存文件数据所在文件的文件名
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
N_Agent=1
M_Enemy=1 #测试为4——————————————————————————————————————————————————————————————————————————————————————————————————————训练为1
RENDER=False  #训练改为False——————————————————————————————————————————————————————————————————————————————————————————————True(可视化)
TRAIN_NUM = 1
TEST_EPIOSDE=100
env = RlGame(n=N_Agent,m=M_Enemy,render=RENDER).unwrapped
state_number=7
action_number=env.action_space.shape[0]
max_action = env.action_space.high[0]
min_action = env.action_space.low[0]
EP_MAX = 1000
EP_LEN = 1000
GAMMA = 0.99
q_lr = 1e-4
value_lr = 3e-4
policy_lr = 3e-4
BATCH = 128
tau = 1e-2
MemoryCapacity=100000
Switch=0 #测试（1）————————————————————————————————————————————————————开关————————————————————————————————————————————————————————————————训练(0)
TOTAL_AGENTS = N_Agent + M_Enemy
STATE_DIM_TOTAL = state_number * TOTAL_AGENTS
ACTION_DIM_TOTAL = action_number * TOTAL_AGENTS
REWARD_DIM_TOTAL = TOTAL_AGENTS
DELAY_UPDATE = 2
TAU_B = 0.01
STD_BIAS = 0.1
HUBER_DELTA = 50.0
TD_BOUND_SCALE = 3.0

def huber_loss_tensor(pred, target, delta=1.0):
    diff = pred - target
    abs_diff = diff.abs()
    quadratic = torch.clamp(abs_diff, max=delta)
    linear = abs_diff - quadratic
    return 0.5 * quadratic.pow(2) + delta * linear

def sample_from_distribution(mean, std):
    noise = torch.randn_like(mean)
    noise = torch.clamp(noise, -3.0, 3.0)
    return mean + noise * std

def compute_target_q(reward, done, current_q, running_std, next_q_mean, next_q_sample, next_log_prob, alpha):
    target_q = reward + (1 - done) * GAMMA * (next_q_mean - alpha * next_log_prob)
    target_q_sample = reward + (1 - done) * GAMMA * (next_q_sample - alpha * next_log_prob)
    td_bound = TD_BOUND_SCALE * running_std
    difference = torch.clamp(target_q_sample - current_q, -td_bound, td_bound)
    target_q_bound = current_q + difference
    return target_q.detach(), target_q_bound.detach()

class Ornstein_Uhlenbeck_Noise:
    def __init__(self, mu, sigma=0.1, theta=0.1, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + \
            self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        '''
        后两行是dXt，其中后两行的前一行是θ(μ-Xt)dt，后一行是σεsqrt(dt)
        '''
        self.x_prev = x
        return x

    def reset(self):
        if self.x0 is not None:
            self.x_prev = self.x0
        else:
            self.x_prev = np.zeros_like(self.mu)

class ActorNet(nn.Module):
    def __init__(self,inp,outp):
        super(ActorNet, self).__init__()
        self.in_to_y1=nn.Linear(inp,256)
        self.in_to_y1.weight.data.normal_(0,0.1)
        self.y1_to_y2=nn.Linear(256,256)
        self.y1_to_y2.weight.data.normal_(0,0.1)
        self.out=nn.Linear(256,outp)
        self.out.weight.data.normal_(0,0.1)
        self.std_out = nn.Linear(256, outp)
        self.std_out.weight.data.normal_(0, 0.1)

    def forward(self,inputstate):
        inputstate=self.in_to_y1(inputstate)
        inputstate=F.relu(inputstate)
        inputstate=self.y1_to_y2(inputstate)
        inputstate=F.relu(inputstate)
        mean=max_action*torch.tanh(self.out(inputstate))#输出概率分布的均值mean
        log_std=self.std_out(inputstate)#softplus激活函数的值域>0
        log_std=torch.clamp(log_std,-20,2)
        std=log_std.exp()
        return mean,std

class CriticNet(nn.Module):
    def __init__(self,input_dim,action_dim):
        super(CriticNet, self).__init__()
        #hidden_size = 256
        hidden_size = 512
        total_input = input_dim + action_dim
        # q1
        self.q1_fc1 = nn.Linear(total_input, hidden_size)
        self.q1_fc2 = nn.Linear(hidden_size, hidden_size)
        self.q1_mean = nn.Linear(hidden_size, 1)
        self.q1_std = nn.Linear(hidden_size, 1)
        # q2
        self.q2_fc1 = nn.Linear(total_input, hidden_size)
        self.q2_fc2 = nn.Linear(hidden_size, hidden_size)
        self.q2_mean = nn.Linear(hidden_size, 1)
        self.q2_std = nn.Linear(hidden_size, 1)

        for layer in [self.q1_fc1, self.q1_fc2, self.q1_mean, self.q1_std,
                      self.q2_fc1, self.q2_fc2, self.q2_mean, self.q2_std]:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(0, 0.1)
                layer.bias.data.zero_()

    def _forward_branch(self, fc1, fc2, mean_head, std_head, inputstate):
        out = F.relu(fc1(inputstate))
        out = F.relu(fc2(out))
        mean = mean_head(out)
        log_std = torch.clamp(std_head(out), min=-5.0, max=2.0)
        std = F.softplus(log_std) + 1e-4
        return mean, std

    def forward(self,s,a):
        inputstate = torch.cat((s, a), dim=1)
        q1_mean, q1_std = self._forward_branch(self.q1_fc1, self.q1_fc2, self.q1_mean, self.q1_std, inputstate)
        q2_mean, q2_std = self._forward_branch(self.q2_fc1, self.q2_fc2, self.q2_mean, self.q2_std, inputstate)
        return q1_mean, q1_std, q2_mean, q2_std

class Memory():
    def __init__(self,capacity,dims):
        self.capacity=capacity
        self.mem=np.zeros((capacity,dims))
        self.memory_counter=0
    '''存储记忆'''
    def store_transition(self,s,a,r,d,s_):
        tran = np.hstack((s, a, r, d, s_))  # 把 s,a,r,done,s_ 直接拼接成一个长向量
        index = self.memory_counter % self.capacity#除余得索引
        self.mem[index, :] = tran  # 给索引存值，第index行所有列都为其中一次的s,a,r,s_；mem会是一个capacity行，（s+a+r+s_）列的数组，再把这条长向量当作矩阵的一行存进去
        self.memory_counter+=1
    '''随机从记忆库里抽取'''
    def sample(self,n):
        assert self.memory_counter>=self.capacity,'记忆库没有存满记忆'
        sample_index = np.random.choice(self.capacity, n)#从capacity个记忆里随机抽取n个为一批，可得到抽样后的索引号
        new_mem = self.mem[sample_index, :]#由抽样得到的索引号在所有的capacity个记忆中  得到记忆s，a，r，s_
        return new_mem

class Actor():
    def __init__(self):
        self.action_net=ActorNet(state_number,action_number)#这只是均值mean
        self.optimizer=torch.optim.Adam(self.action_net.parameters(),lr=policy_lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=EP_MAX, eta_min=1e-5) #余弦退火调度器
    
    def step_scheduler(self):
        self.scheduler.step()
        
    def choose_action(self,s):
        if not torch.is_tensor(s):
            inputstate = torch.FloatTensor(s).unsqueeze(0)
        else:
            inputstate = s.unsqueeze(0) if s.dim() == 1 else s
        mean,std=self.action_net(inputstate)
        dist = torch.distributions.Normal(mean, std)
        action = torch.tanh(dist.sample())
        action=torch.clamp(action,min_action,max_action)
        return action.squeeze(0).detach().numpy()
    def evaluate(self,s):
        if not torch.is_tensor(s):
            inputstate = torch.FloatTensor(s)
        else:
            inputstate = s
        if inputstate.dim() == 1:
            inputstate = inputstate.unsqueeze(0)
        mean,std=self.action_net(inputstate)
        dist = torch.distributions.Normal(mean, std)
        noise = torch.randn_like(mean)
        pre_tanh = mean + std * noise
        action=torch.tanh(pre_tanh)
        action=torch.clamp(action,min_action,max_action)
        log_prob=dist.log_prob(pre_tanh)-torch.log(1-action.pow(2)+1e-6)
        if log_prob.dim() == 2:
            log_prob = log_prob.sum(dim=1, keepdim=True)
        else:
            log_prob = log_prob.sum().unsqueeze(0).unsqueeze(0)
        return action,log_prob

    def learn(self,actor_loss):
        loss=actor_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Entroy():
    def __init__(self):
        self.target_entropy = -action_number
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.log_alpha], lr=q_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def learn(self,entroy_loss):
        loss=entroy_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Critic():
    def __init__(self):
        self.critic_v = CriticNet(STATE_DIM_TOTAL, action_number)#改网络输入状态，生成一个Q值
        self.target_critic_v = CriticNet(STATE_DIM_TOTAL, action_number)
        self.target_critic_v.load_state_dict(self.critic_v.state_dict())
        self.optimizer = torch.optim.Adam(self.critic_v.parameters(), lr=value_lr,eps=1e-5)
        self.mean_std1 = None
        self.mean_std2 = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def soft_update(self):
        for target_param, param in zip(self.target_critic_v.parameters(), self.critic_v.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def get_v(self,s,a):
        return self.critic_v(s,a)

    def target_get_v(self,s,a):
        return self.target_critic_v(s,a)

def main():
    run(env)
def run(env):
    if Switch==0:
        try:
            assert M_Enemy == 1
        except:
            print('程序终止，被逮到~嘿嘿，哥们儿预判到你会犯错，这段程序中变量\'M_Enemy\'的值必须为1，请把它的值改为1。\n' 
                  '改为1之后程序一定会报错，这是因为组数越界，更改path_env.py文件中的跟随者无人机初始化个数；删除多余的\n'
                  '求距离函数，即变量dis_1_agent_0_to_3等，以及提到变量dis_1_agent_0_to_3等的地方；删除画无人机轨迹的\n'
                  '函数；删除step函数的最后一个返回值dis_1_agent_0_to_1；将player.py文件中的变量dt改为1；即可开始训练！\n'
                  '如果实在不会改也无妨，我会在不久之后出一个视频来手把手教大伙怎么改，可持续关注此项目github中的README文件。\n')
        else:
            print('DSACT训练中...')
            all_ep_r = [[] for i in range(TRAIN_NUM)]
            all_ep_r0 = [[] for i in range(TRAIN_NUM)]
            all_ep_r1 = [[] for i in range(TRAIN_NUM)]
            for k in range(TRAIN_NUM):
                actors = [None for _ in range(N_Agent+M_Enemy)]
                critics = [None for _ in range(N_Agent+M_Enemy)]
                entroys = [None for _ in range(N_Agent+M_Enemy)]
                update_step = 0
                for i in range(N_Agent+M_Enemy):
                    actors[i] = Actor()
                    critics[i] = Critic()
                    entroys[i] = Entroy()
                mem_dims = 2 * STATE_DIM_TOTAL + ACTION_DIM_TOTAL + 2 * TOTAL_AGENTS
                M = Memory(MemoryCapacity, mem_dims)
                ou_noise = Ornstein_Uhlenbeck_Noise(mu=np.zeros(((N_Agent+M_Enemy), action_number)))
                action=np.zeros(((N_Agent+M_Enemy), action_number))
                # aaa = np.zeros((N_Agent, state_number))
                for episode in range(EP_MAX):
                    observation = env.reset()  # 环境重置
                    reward_totle,reward_totle0,reward_totle1 = 0,0,0
                    for timestep in range(EP_LEN):
                        for i in range(N_Agent+M_Enemy):
                            action[i] = actors[i].choose_action(observation[i])
                        # action[0]=actor0.choose_action(observation[0])
                        # action[1] = actor0.choose_action(observation[1])
                        if episode <= 20:
                            noise = ou_noise()
                        else:
                            noise = 0
                        action = action + noise
                        action = np.clip(action, -max_action, max_action)
                        observation_, reward,done,win,team_counter= env.step(action)  # 单步交互
                        done_array = np.full((TOTAL_AGENTS,), float(done))
                        M.store_transition(observation.flatten(), action.flatten(), reward.flatten(), done_array.flatten(), observation_.flatten())
                        # 记忆库存储
                        # 有的2000个存储数据就开始学习
                        if M.memory_counter > MemoryCapacity:
                            b_M = M.sample(BATCH)
                            b_s = b_M[:, :STATE_DIM_TOTAL]
                            b_a = b_M[:, STATE_DIM_TOTAL: STATE_DIM_TOTAL + ACTION_DIM_TOTAL]
                            reward_start = STATE_DIM_TOTAL + ACTION_DIM_TOTAL
                            reward_end = reward_start + TOTAL_AGENTS
                            done_end = reward_end + TOTAL_AGENTS
                            b_r = b_M[:, reward_start:reward_end]
                            b_done = b_M[:, reward_end:done_end]
                            b_s_ = b_M[:, done_end:]
                            b_s = torch.FloatTensor(b_s)
                            b_a = torch.FloatTensor(b_a)
                            b_r = torch.FloatTensor(b_r)
                            b_done = torch.FloatTensor(b_done)
                            b_s_ = torch.FloatTensor(b_s_)
                            update_step += 1
                            update_actor_flag = (update_step % DELAY_UPDATE == 0)
                            for i in range(N_Agent+M_Enemy):
                                critic = critics[i]
                                actor = actors[i]
                                entroy = entroys[i]
                                state_slice = b_s[:, state_number*i:state_number*(i+1)]
                                next_state_slice = b_s_[:, state_number*i:state_number*(i+1)]
                                action_slice = b_a[:, action_number*i:action_number*(i+1)]
                                reward_slice = b_r[:, i:(i+1)]
                                done_slice = b_done[:, i:(i+1)]

                                new_action_next, log_prob_next = actor.evaluate(next_state_slice)
                                q1_next_mean, q1_next_std, q2_next_mean, q2_next_std = critic.target_get_v(b_s_, new_action_next)
                                q1_next_sample = sample_from_distribution(q1_next_mean, q1_next_std)
                                q2_next_sample = sample_from_distribution(q2_next_mean, q2_next_std)
                                q_next_mean = torch.min(q1_next_mean, q2_next_mean)
                                q_next_sample = torch.where(q1_next_mean < q2_next_mean, q1_next_sample, q2_next_sample)

                                q1_mean, q1_std, q2_mean, q2_std = critic.get_v(b_s, action_slice)
                                std1_mean = torch.mean(q1_std.detach())
                                std2_mean = torch.mean(q2_std.detach())
                                if critic.mean_std1 is None:
                                    critic.mean_std1 = std1_mean
                                else:
                                    critic.mean_std1 = (1 - TAU_B) * critic.mean_std1 + TAU_B * std1_mean
                                if critic.mean_std2 is None:
                                    critic.mean_std2 = std2_mean
                                else:
                                    critic.mean_std2 = (1 - TAU_B) * critic.mean_std2 + TAU_B * std2_mean

                                alpha_value = entroy.alpha.detach()
                                target_q1, target_q1_bound = compute_target_q(
                                    reward_slice,
                                    done_slice,
                                    q1_mean.detach(),
                                    critic.mean_std1.detach(),
                                    q_next_mean.detach(),
                                    q_next_sample.detach(),
                                    log_prob_next.detach(),
                                    alpha_value,
                                )
                                target_q2, target_q2_bound = compute_target_q(
                                    reward_slice,
                                    done_slice,
                                    q2_mean.detach(),
                                    critic.mean_std2.detach(),
                                    q_next_mean.detach(),
                                    q_next_sample.detach(),
                                    log_prob_next.detach(),
                                    alpha_value,
                                )

                                q1_std_detach = torch.clamp(q1_std, min=0.).detach()
                                q2_std_detach = torch.clamp(q2_std, min=0.).detach()
                                ratio1 = ((critic.mean_std1.detach() ** 2) / (torch.pow(q1_std_detach, 2) + STD_BIAS)).clamp(min=0.1, max=10)
                                ratio2 = ((critic.mean_std2.detach() ** 2) / (torch.pow(q2_std_detach, 2) + STD_BIAS)).clamp(min=0.1, max=10)

                                q1_loss = ratio1 * (
                                        huber_loss_tensor(q1_mean, target_q1, HUBER_DELTA)
                                        + q1_std * (
                                                torch.pow(q1_std_detach, 2)
                                                - huber_loss_tensor(q1_mean.detach(), target_q1_bound, HUBER_DELTA)
                                        ) / (q1_std_detach + STD_BIAS)
                                )
                                q2_loss = ratio2 * (
                                        huber_loss_tensor(q2_mean, target_q2, HUBER_DELTA)
                                        + q2_std * (
                                                torch.pow(q2_std_detach, 2)
                                                - huber_loss_tensor(q2_mean.detach(), target_q2_bound, HUBER_DELTA)
                                        ) / (q2_std_detach + STD_BIAS)
                                )

                                critic_loss = (q1_loss.mean() + q2_loss.mean())
                                critic.optimizer.zero_grad()
                                critic_loss.backward()
                                critic.optimizer.step()

                                if update_actor_flag:
                                    new_action, log_prob = actor.evaluate(state_slice)
                                    q1_pi, _, q2_pi, _ = critic.get_v(b_s, new_action)
                                    q_pi = torch.min(q1_pi, q2_pi)
                                    actor_loss = (alpha_value * log_prob - q_pi).mean()
                                    actor.learn(actor_loss)
                                    alpha_loss = -(entroy.log_alpha * (log_prob.detach() + entroy.target_entropy)).mean()
                                    entroy.learn(alpha_loss)

                            if update_actor_flag:
                                for critic in critics:
                                    critic.soft_update()
                        observation = observation_
                        reward_totle += reward.mean()
                        reward_totle0 += float(reward[0])
                        reward_totle1 += float(reward[1])
                        if RENDER:
                            env.render()
                        if done:
                            break
                    print("Ep: {} rewards: {}".format(episode, reward_totle))
                    all_ep_r[k].append(reward_totle)
                    all_ep_r0[k].append(reward_totle0)
                    all_ep_r1[k].append(reward_totle1)
                    if episode % 20 == 0 and episode > 200:#保存神经网络参数
                        save_data = {'net': actors[0].action_net.state_dict(), 'opt': actors[0].optimizer.state_dict()}
                        torch.save(save_data, "E:/PycharmProjects/code/backup/path planning/Path_DSACT_actor_L1.pth")
                        save_data = {'net': actors[1].action_net.state_dict(), 'opt': actors[1].optimizer.state_dict()}
                        torch.save(save_data, "E:/PycharmProjects/code/backup/path planning/Path_DSACT_actor_F1.pth")
            all_ep_r_mean = np.mean((np.array(all_ep_r)), axis=0)
            all_ep_r_std = np.std((np.array(all_ep_r)), axis=0)
            all_ep_L_mean = np.mean((np.array(all_ep_r0)), axis=0)
            all_ep_L_std = np.std((np.array(all_ep_r0)), axis=0)
            all_ep_F_mean = np.mean((np.array(all_ep_r1)), axis=0)
            all_ep_F_std = np.std((np.array(all_ep_r1)), axis=0)
            d = {"all_ep_r_mean": all_ep_r_mean, "all_ep_r_std": all_ep_r_std,
                 "all_ep_L_mean": all_ep_L_mean, "all_ep_L_std": all_ep_L_std,
                 "all_ep_F_mean": all_ep_F_mean, "all_ep_F_std": all_ep_F_std,}
            f = open(shoplistfile, 'wb')  # 二进制打开，如果找不到该文件，则创建一个
            pkl.dump(d, f, pkl.HIGHEST_PROTOCOL)  # 写入文件
            f.close()
            all_ep_r_max = all_ep_r_mean + all_ep_r_std * 0.95
            all_ep_r_min = all_ep_r_mean - all_ep_r_std * 0.95
            all_ep_L_max = all_ep_L_mean + all_ep_L_std * 0.95
            all_ep_L_min = all_ep_L_mean - all_ep_L_std * 0.95
            all_ep_F_max = all_ep_F_mean + all_ep_F_std * 0.95
            all_ep_F_min = all_ep_F_mean - all_ep_F_std * 0.95
            plt.margins(x=0)
            plt.plot(np.arange(len(all_ep_r_mean)), all_ep_r_mean, label='DSACT', color='#e75840')
            plt.fill_between(np.arange(len(all_ep_r_mean)), all_ep_r_max, all_ep_r_min, alpha=0.6, facecolor='#e75840')
            plt.xlabel('Episode')
            plt.ylabel('Total reward')   #最后训练结束图像
            plt.figure(2, figsize=(8, 4), dpi=150)
            plt.margins(x=0)
            plt.plot(np.arange(len(all_ep_L_mean)), all_ep_L_mean, label='DSACT', color='#e75840')
            plt.fill_between(np.arange(len(all_ep_L_mean)), all_ep_L_max, all_ep_L_min, alpha=0.6,
                             facecolor='#e75840')
            plt.xlabel('Episode')
            plt.ylabel('Leader reward')
            plt.figure(3, figsize=(8, 4), dpi=150)
            plt.margins(x=0)
            plt.plot(np.arange(len(all_ep_F_mean)), all_ep_F_mean, label='DSACT', color='#e75840')
            plt.fill_between(np.arange(len(all_ep_F_mean)), all_ep_F_max, all_ep_F_min, alpha=0.6,
                             facecolor='#e75840')
            plt.xlabel('Episode')
            plt.ylabel('Follower reward')
            plt.legend()
            plt.show()
            env.close()
    else:
         print('测试在main_SAC_dsact_test.py中进行，请运行该文件。')
         
if __name__ == '__main__':
    main()

