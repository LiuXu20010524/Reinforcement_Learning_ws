import torch
from torch import nn
import numpy as np
import pandas as pd
import PIL
import glob
import math
import os
import torchvision
import gym
#from pyvirtualdisplay import Display
import matplotlib.pyplot as plt
import copy
import time
import random
import mujoco
from scipy import stats

#创建actor网络
class actor_net(nn.Module):
    
    def __init__(self,N_S,N_A):
        super().__init__()
        self.linear1=nn.Linear(N_S,64)
        self.linear2=nn.Linear(64,128)
        self.linear3=nn.Linear(128,64)
        self.mean=nn.Linear(64,N_A)
        self.vair=nn.Linear(64,N_A)
    
    def forward(self,inputs):
        x=self.linear1(inputs)
        x=torch.relu(x)
        x=self.linear2(x)
        x=torch.relu(x)
        x=self.linear3(x)
        x=torch.relu(x)
        mean=torch.tanh(self.mean(x))
        vair=torch.exp(torch.tanh(self.vair(x))-2)
        return mean,vair
    
    def choose_data(self,s):
        mu,va=self.forward(s)
        Pi=torch.distributions.Normal(mu,va)
        return Pi.sample().numpy()
#创建critic网路
class critic_net(nn.Module):
    
    def __init__(self,N_S):
        super().__init__()
        self.linear1=nn.Linear(N_S,64)
        self.linear2=nn.Linear(64,128)
        self.linear3=nn.Linear(128,64)
        self.out=nn.Linear(64,1)
    
    def forward(self,inputs):
        x=torch.relu(self.linear1(inputs))
        x=torch.relu(self.linear2(x))
        x=torch.relu(self.linear3(x))
        out=self.out(x)
        return out
actor=actor_net(27,8)
critic=critic_net(27)
actor.load_state_dict(torch.load('./actor_state_dict.pth'))
critic.load_state_dict(torch.load('./critic_state_dict.pth'))
# env=gym.make("Ant-v4",exclude_current_positions_from_observation=False,render_mode='rgb_array')
env=gym.make("Ant-v4",healthy_reward=0,ctrl_cost_weight=0,render_mode='human')
observation,info=env.reset()
times=0
while(1):
    env.render()
    with torch.no_grad():
        actionxxx=actor.choose_data(torch.from_numpy(observation).type(torch.FloatTensor))
    observation, reward, truncated, terminated, info=env.step(actionxxx)
    times+=1
    if terminated:
        print('end by 1000,times=',times)
        print(info)
        break
    elif truncated:
        print('end by fail,times=',times)
        print(info)
        break
env.close()
print('end')
