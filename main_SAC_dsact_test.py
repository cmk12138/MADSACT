# -*- coding: utf-8 -*-
#开发者：Bright Fang
#开发时间：2023/7/30 18:13
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
RENDER=True  #训练改为False——————————————————————————————————————————————————————————————————————————————————————————————True(可视化)
TRAIN_NUM = 1
TEST_EPIOSDE=100
env = RlGame(n=N_Agent,m=M_Enemy,render=RENDER).unwrapped
state_number=7
action_number=env.action_space.shape[0]
max_action = env.action_space.high[0]
min_action = env.action_space.low[0]
EP_MAX = 500
EP_LEN = 1000
GAMMA = 0.9
q_lr = 3e-4
value_lr = 3e-3
policy_lr = 1e-3
BATCH = 128
tau = 1e-2
MemoryCapacity=20000
Switch=1#测试（1）————————————————————————————————————————————————————开关————————————————————————————————————————————————————————————————训练(0)

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
    
class Actor():
    def __init__(self):
        self.action_net=ActorNet(state_number,action_number)#这只是均值mean
        self.optimizer=torch.optim.Adam(self.action_net.parameters(),lr=policy_lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=EP_MAX, eta_min=1e-5)
        
    def step_scheduler(self):
        self.scheduler.step()

    def choose_action(self,s):
        inputstate = torch.FloatTensor(s)
        mean,std=self.action_net(inputstate)
        dist = torch.distributions.Normal(mean, std)
        action=dist.sample()
        action=torch.clamp(action,min_action,max_action)
        return action.detach().numpy()
    def evaluate(self,s):
        inputstate = torch.FloatTensor(s)
        mean,std=self.action_net(inputstate)
        dist = torch.distributions.Normal(mean, std)
        noise = torch.distributions.Normal(0, 1)
        z = noise.sample()
        action=torch.tanh(mean+std*z)
        action=torch.clamp(action,min_action,max_action)
        action_logprob=dist.log_prob(mean+std*z)-torch.log(1-action.pow(2)+1e-6)
        return action,action_logprob

    def learn(self,actor_loss):
        loss=actor_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
def main():
    run(env)
def run(env):
    if Switch==0:
        print('训练在main_SAC_dsact.py中进行，请运行该文件。')
    else:
        print('DSACT测试中...')
        aa = Actor()
        checkpoint_aa = torch.load("E:/PycharmProjects/code/backup/path planning/Path_DSACT_actor_L1.pth")
        aa.action_net.load_state_dict(checkpoint_aa['net'])
        bb = Actor()
        checkpoint_bb = torch.load("E:/PycharmProjects/code/backup/path planning/Path_DSACT_actor_F1.pth")
        bb.action_net.load_state_dict(checkpoint_bb['net'])
        action = np.zeros((N_Agent+M_Enemy, action_number))
        win_times = 0
        average_FKR=0
        average_timestep=0
        average_integral_V=0
        average_integral_U= 0
        all_ep_V, all_ep_U, all_ep_T, all_ep_F = [], [], [], []
        for j in range(TEST_EPIOSDE):
            state = env.reset()
            total_rewards = 0
            integral_V=0
            integral_U=0
            v,v1,Dis=[],[],[]
            for timestep in range(EP_LEN):
                for i in range(N_Agent):
                    action[i] = aa.choose_action(state[i])
                for i in range(M_Enemy):
                    action[i+1] = bb.choose_action(state[i+1])
                new_state, reward,done,win,team_counter,dis = env.step(action)  # 执行动作
                if win:
                    win_times += 1
                v.append(state[0][2]*30)
                v1.append(state[1][2]*30)
                Dis.append(dis)
                integral_V+=state[0][2]
                integral_U+=abs(action[0]).sum()
                total_rewards += reward.mean()
                state = new_state
                if RENDER:
                    env.render()
                if done:
                    break
            FKR=team_counter/timestep
            average_FKR += FKR
            average_timestep += timestep
            average_integral_V += integral_V
            average_integral_U += integral_U
            print("Score", total_rewards)
            print("Epiosde", j)
            all_ep_V.append(integral_V)
            all_ep_U.append(integral_U)
            all_ep_T.append(timestep)
            all_ep_F.append(FKR)
        print('任务完成率',win_times / TEST_EPIOSDE)
        print('平均最大编队保持率', average_FKR/TEST_EPIOSDE)
        print('平均最短飞行时间', average_timestep/TEST_EPIOSDE)
        print('平均最短飞行路程', average_integral_V/TEST_EPIOSDE)
        print('平均最小能量损耗', average_integral_U/TEST_EPIOSDE)
        env.close()

if __name__ == '__main__':
    main()
